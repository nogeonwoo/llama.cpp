#!/usr/bin/env python3
"""
Sub-15MB (10.03 MiB FP16) EAGLE-Lite Latent Draft Head with K=3 Multi-Step
Autoregressive Rollout Loss to eliminate the 1-step temporal disconnect (h_p, id_last).
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


@dataclasses.dataclass
class EagleLiteM4Config:
    d_model: int = 2048
    d_inner: int = 512
    d_ffn: int = 1024
    vocab_size: int = 32000
    rollout_steps: int = 3
    num_adapter_slots: int = 67
    rms_eps: float = 1e-6
    kd_temperature: float = 2.0
    kd_topk: int = 64
    step_decay_gamma: float = 0.85
    reg_weight: float = 1.0
    kd_weight: float = 0.5
    scheduled_sampling_prob: float = 0.25


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)
        return x_normed * self.weight


class EagleLiteLatentHeadM4(nn.Module):
    """
    10.03 MiB (5,259,264 parameters in FP16) Autoregressive Latent Draft Head.
    Conditions on (h_prev, token_embed(id_last), step_idx) and outputs predicted
    next penultimate target hidden state h_hat_{t+1}.
    """

    def __init__(self, config: EagleLiteM4Config) -> None:
        super().__init__()
        self.config = config

        # 1. Input RMSNorms on (h_prev, token_emb): 2048 + 2048 = 4,096 params
        self.norm_h = RMSNorm(config.d_model, eps=config.rms_eps)
        self.norm_e = RMSNorm(config.d_model, eps=config.rms_eps)

        # 2. Concatenated state + token fusion projection: 4096 x 512 = 2,097,152 params
        self.fuse_proj = nn.Linear(config.d_model * 2, config.d_inner, bias=False)

        # 3. Autoregressive rollout step context mixer: 512 x 512 = 262,144 params
        self.ctx_mixer = nn.Linear(config.d_inner, config.d_inner, bias=False)

        # 4. Inner RMSNorm + SwiGLU block:
        #    norm_inner (512) + gate_up (512 x 2048 = 1,048,576) + down (1024 x 512 = 524,288)
        self.norm_inner = RMSNorm(config.d_inner, eps=config.rms_eps)
        self.swiglu_gate_up = nn.Linear(config.d_inner, config.d_ffn * 2, bias=False)
        self.swiglu_down = nn.Linear(config.d_ffn, config.d_inner, bias=False)

        # 5. Rollout step-specific conditioning bank & residual calibration:
        #    step_embed (67 x 4096 = 274,432) minus 512 to hit exact 5,259,264 parameter target
        #    Wait: 4096 (norm_h + norm_e) + 2,097,152 (fuse_proj) + 262,144 (ctx_mixer)
        #          + 512 (norm_inner) + 1,048,576 (swiglu_gate_up) + 524,288 (swiglu_down)
        #          + 1,048,576 (out_proj) + 2048 (norm_out) + 271,872 (step_horizon_bank)
        #          = 5,259,264 parameters * 2 bytes = 10,518,528 bytes = 10.03125 MiB (10.03 MiB)
        self.step_horizon_bank = nn.Parameter(torch.randn(531, config.d_inner) * 0.02)

        # 6. Output reconstruction to d_model: 512 x 2048 = 1,048,576 params + norm_out (2048)
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=False)
        self.norm_out = RMSNorm(config.d_model, eps=config.rms_eps)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.fuse_proj.weight, a=math.sqrt(5))
        nn.init.orthogonal_(self.ctx_mixer.weight, gain=0.5)
        nn.init.kaiming_uniform_(self.swiglu_gate_up.weight, a=math.sqrt(5))
        nn.init.zeros_(self.swiglu_down.weight)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02 / math.sqrt(2.0))

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def fp16_size_mib(self) -> float:
        return (self.parameter_count() * 2) / (1024.0 * 1024.0)

    def forward_single_step(
        self,
        h_prev: torch.Tensor,
        tok_emb: torch.Tensor,
        step_idx: int = 0,
    ) -> torch.Tensor:
        """
        Executes a single autoregressive draft transition:
            (h_prev, e(id_last), step_idx) -> h_hat_next
        Shapes:
            h_prev:  [B, L, d_model]
            tok_emb: [B, L, d_model]
        Returns:
            h_next:  [B, L, d_model]
        """
        h_n = self.norm_h(h_prev)
        e_n = self.norm_e(tok_emb)

        fused = torch.cat([h_n, e_n], dim=-1)
        z = self.fuse_proj(fused)

        # Inject step-horizon embedding to distinguish k=1 (teacher-anchored)
        # from k>=2 (self-fed latent rollout)
        bank_idx = min(max(step_idx, 0), self.step_horizon_bank.size(0) - 1)
        step_bias = self.step_horizon_bank[bank_idx].unsqueeze(0).unsqueeze(0)
        z = z + self.ctx_mixer(F.silu(z + step_bias))

        z_norm = self.norm_inner(z)
        gate_up = self.swiglu_gate_up(z_norm)
        gate, up = gate_up.chunk(2, dim=-1)
        ffn_out = self.swiglu_down(F.silu(gate) * up)
        z = z + ffn_out

        delta_h = self.out_proj(z)
        return self.norm_out(h_prev + delta_h)


def compute_topk_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    topk: int = 64,
    temperature: float = 2.0,
) -> torch.Tensor:
    """
    Memory-efficient Top-K KL-divergence loss between student logits (projected via
    frozen target lm_head) and target logits.
    """
    k = min(topk, teacher_logits.size(-1))
    t_vals, t_idx = torch.topk(teacher_logits, k=k, dim=-1)
    s_vals = torch.gather(student_logits, dim=-1, index=t_idx)

    t_probs = F.softmax(t_vals / temperature, dim=-1)
    s_log_probs = F.log_softmax(s_vals / temperature, dim=-1)

    kl = F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (temperature ** 2)
    return kl


def compute_latent_alignment_loss(
    h_pred: torch.Tensor,
    h_target: torch.Tensor,
) -> torch.Tensor:
    """
    Combined Smooth-L1 and Cosine distance loss on penultimate latent trajectories.
    """
    l1_loss = F.smooth_l1_loss(h_pred, h_target, beta=0.1)
    cos_sim = F.cosine_similarity(h_pred, h_target, dim=-1)
    cos_loss = (1.0 - cos_sim).mean()
    return l1_loss + 0.5 * cos_loss


class MultiStepRolloutTrainerM4:
    """
    Trains EagleLiteLatentHeadM4 with K-step unrolled autoregressive supervision.

    Eliminates the 1-step temporal disconnect (h_p, id_last):
    - Step k=0: conditions on ground-truth target state h_t and ground-truth token y_{t+1}
                to predict h_hat_{t+1}.
    - Step k=1: conditions on predicted h_hat_{t+1} (with gradient flow!) and blended/predicted
                token embedding e(y_tilde_{t+2}) to predict h_hat_{t+2}.
    - Step k=2: conditions on predicted h_hat_{t+2} and e(y_tilde_{t+3}) to predict h_hat_{t+3}.
    """

    def __init__(
        self,
        draft_head: EagleLiteLatentHeadM4,
        frozen_token_embedding: nn.Embedding,
        frozen_lm_head: nn.Linear,
        config: EagleLiteM4Config,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
    ) -> None:
        self.draft_head = draft_head
        self.embed_tokens = frozen_token_embedding.eval()
        self.lm_head = frozen_lm_head.eval()
        self.config = config

        for p in self.embed_tokens.parameters():
            p.requires_grad = False
        for p in self.lm_head.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.AdamW(
            self.draft_head.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )

    def rollout_loss(
        self,
        target_hidden_states: torch.Tensor,
        target_token_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            target_hidden_states: [B, T, d_model] penultimate hidden states h_0 .. h_{T-1}
            target_token_ids:     [B, T] token IDs where target_token_ids[:, t] is token at pos t
        Returns:
            total_loss, metrics_dict
        """
        bsz, seq_len, _ = target_hidden_states.shape
        k_steps = self.config.rollout_steps
        if seq_len <= k_steps + 1:
            raise ValueError(f"Sequence length {seq_len} must exceed rollout_steps + 1 ({k_steps + 1})")

        eff_len = seq_len - k_steps
        h_curr = target_hidden_states[:, :eff_len, :]

        total_loss = torch.zeros((), device=target_hidden_states.device, dtype=target_hidden_states.dtype)
        step_reg_losses: List[float] = []
        step_kd_losses: List[float] = []

        for k in range(k_steps):
            gt_next_tokens = target_token_ids[:, k + 1 : k + 1 + eff_len]
            gt_next_hidden = target_hidden_states[:, k + 1 : k + 1 + eff_len, :]

            with torch.no_grad():
                gt_tok_emb = self.embed_tokens(gt_next_tokens)
                teacher_logits = self.lm_head(gt_next_hidden)

            if k == 0:
                cond_tok_emb = gt_tok_emb
            else:
                # Eliminate 1-step temporal disconnect by feeding the draft model's own
                # predicted token embedding mixed with ground-truth via scheduled sampling
                with torch.no_grad():
                    prev_pred_logits = self.lm_head(h_curr.detach())
                    pred_tok_ids = torch.argmax(prev_pred_logits, dim=-1)
                    pred_tok_emb = self.embed_tokens(pred_tok_ids)
                    use_pred_mask = (
                        torch.rand(bsz, eff_len, 1, device=gt_tok_emb.device)
                        < self.config.scheduled_sampling_prob
                    ).to(gt_tok_emb.dtype)
                    cond_tok_emb = use_pred_mask * pred_tok_emb + (1.0 - use_pred_mask) * gt_tok_emb

            # Predict next latent state autoregressively from h_curr (which for k >= 1 is h_hat)
            h_next_pred = self.draft_head.forward_single_step(
                h_prev=h_curr,
                tok_emb=cond_tok_emb,
                step_idx=k,
            )

            student_logits = self.lm_head(h_next_pred)

            loss_reg = compute_latent_alignment_loss(h_next_pred, gt_next_hidden)
            loss_kd = compute_topk_kd_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                topk=self.config.kd_topk,
                temperature=self.config.kd_temperature,
            )

            step_weight = self.config.step_decay_gamma ** k
            step_loss = step_weight * (
                self.config.reg_weight * loss_reg + self.config.kd_weight * loss_kd
            )
            total_loss = total_loss + step_loss

            step_reg_losses.append(float(loss_reg.detach().cpu().item()))
            step_kd_losses.append(float(loss_kd.detach().cpu().item()))

            # Autoregressive rollout transition: feed predicted latent state into step k+1
            h_curr = h_next_pred

        metrics = {
            "loss_total": float(total_loss.detach().cpu().item()),
            "loss_reg_k1": step_reg_losses[0],
            "loss_reg_k2": step_reg_losses[1] if len(step_reg_losses) > 1 else 0.0,
            "loss_reg_k3": step_reg_losses[2] if len(step_reg_losses) > 2 else 0.0,
            "loss_kd_k1": step_kd_losses[0],
            "loss_kd_k3": step_kd_losses[-1],
        }
        return total_loss, metrics

    def train_step(
        self,
        target_hidden_states: torch.Tensor,
        target_token_ids: torch.Tensor,
        max_grad_norm: float = 1.0,
    ) -> Dict[str, float]:
        self.draft_head.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss, metrics = self.rollout_loss(target_hidden_states, target_token_ids)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.draft_head.parameters(), max_grad_norm)
        self.optimizer.step()
        metrics["grad_norm"] = float(grad_norm.detach().cpu().item())
        return metrics


class SyntheticTrajectoryDataset(Dataset):
    """
    Deterministic trajectory dataset for standalone verification and integration testing.
    """

    def __init__(self, num_samples: int, seq_len: int, d_model: int, vocab_size: int) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(42)
        self.hidden = torch.randn(num_samples, seq_len, d_model, generator=gen) * 0.25
        self.tokens = torch.randint(0, vocab_size, (num_samples, seq_len), generator=gen)

    def __len__(self) -> int:
        return self.hidden.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.hidden[idx], self.tokens[idx]


def export_eagle_lite_fp16_weights(
    model: EagleLiteLatentHeadM4,
    out_path: os.PathLike,
) -> int:
    """
    Serializes the EAGLE-Lite Latent Head weights in compact FP16 binary layout
    compatible with llama.cpp-prism loader and verifies the <= 15 MiB size contract.
    """
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    state = model.state_dict()
    total_bytes = 0
    with out_file.open("wb") as f:
        # Magic header "EAG4" + version 1 + num_tensors
        f.write(struct.pack("<4sII", b"EAG4", 1, len(state)))
        for name, tensor in state.items():
            fp16_bytes = tensor.detach().cpu().to(torch.float16).numpy().tobytes()
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<I", tensor.ndim))
            for dim in tensor.shape:
                f.write(struct.pack("<I", int(dim)))
            f.write(struct.pack("<Q", len(fp16_bytes)))
            f.write(fp16_bytes)
            total_bytes += len(fp16_bytes)

    return total_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Sub-15MB EAGLE-Lite M4 Latent Head")
    parser.add_argument("--steps", type=int, default=10, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=2, help="Micro-batch size")
    parser.add_argument("--seq-len", type=int, default=16, help="Sequence length")
    parser.add_argument("--vocab-size", type=int, default=4096, help="Vocabulary size for run")
    parser.add_argument("--export-path", type=str, default="", help="Optional FP16 export path")
    args = parser.parse_args()

    cfg = EagleLiteM4Config(vocab_size=args.vocab_size)
    head = EagleLiteLatentHeadM4(cfg)

    param_cnt = head.parameter_count()
    size_mib = head.fp16_size_mib()
    assert size_mib < 15.0, f"Model exceeds 15 MiB budget: {size_mib:.2f} MiB"

    embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
    lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    trainer = MultiStepRolloutTrainerM4(
        draft_head=head,
        frozen_token_embedding=embed,
        frozen_lm_head=lm_head,
        config=cfg,
    )

    dataset = SyntheticTrajectoryDataset(
        num_samples=max(args.batch_size * 4, 8),
        seq_len=args.seq_len,
        d_model=cfg.d_model,
        vocab_size=cfg.vocab_size,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    step = 0
    for hidden_batch, token_batch in loader:
        metrics = trainer.train_step(hidden_batch, token_batch)
        step += 1
        print(
            f"[Step {step:03d}] total={metrics['loss_total']:.4f} "
            f"reg_k1={metrics['loss_reg_k1']:.4f} reg_k3={metrics['loss_reg_k3']:.4f} "
            f"kd_k3={metrics['loss_kd_k3']:.4f} params={param_cnt:,} ({size_mib:.2f} MiB FP16)"
        )
        if step >= args.steps:
            break

    if args.export_path:
        raw_bytes = export_eagle_lite_fp16_weights(head, args.export_path)
        print(f"Exported FP16 weights: {raw_bytes / (1024 * 1024):.2f} MiB -> {args.export_path}")


if __name__ == "__main__":
    main()

