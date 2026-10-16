#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#if __has_include("llama.h")
#include "llama.h"
#else
// Standalone compatibility shim when compiled in isolation outside llama.cpp include tree
using llama_token  = int32_t;
using llama_pos    = int32_t;
using llama_seq_id = int32_t;

struct llama_batch {
    int32_t        n_tokens = 0;
    llama_token  * token    = nullptr;
    float        * embd     = nullptr;
    llama_pos    * pos      = nullptr;
    int32_t      * n_seq_id = nullptr;
    llama_seq_id ** seq_id  = nullptr;
    int8_t       * logits   = nullptr;
};

struct llama_context;
#endif

namespace prism::m4 {

// Maximum micro-batch size before ggml-metal transitions from simdgroup specialized
// MV kernels to heavier threadgroup GEMM tiles on Apple Silicon M-series GPUs.
constexpr int32_t kMaxSpecBatchM4 = 4;
constexpr int32_t kMaxSpecDraftM4 = 3;

enum class tree_topology_m4 : uint8_t {
    LINEAR_K3   = 0, // [0: t0] -> [1: d1] -> [2: d2] -> [3: d3]
    BRANCHED_Y1 = 1, // [0: t0] -> [1: d1_a] -> [3: d2_a], and [0: t0] -> [2: d1_b]
    FORKED_Y2   = 2  // [0: t0] -> [1: d1] -> [2: d2_a], and [1: d1] -> [3: d2_b]
};

// [RESUME FROM LINE 975: struct tree_batch_plan_m4]
struct tree_batch_plan_m4 {
    tree_topology_m4           topology   = tree_topology_m4::LINEAR_K3;
    std::array<int32_t, 4>     tokens     = {0, 0, 0, 0};
    std::array<int32_t, 4>     rel_pos    = {0, 1, 2, 3};
    std::array<int32_t, 4>     parent_idx = {-1, 0, 1, 2};
    // Bitmask of active sequence streams (bit 0 = primary branch seq_base + 0,
    // bit 1 = secondary branch seq_base + 1) assigned to each batch token slot.
    std::array<uint8_t, 4>     seq_mask   = {0x01, 0x01, 0x01, 0x01};
    // Explicit 4x4 causal visibility matrix: causal_mask[row][col] == 1 iff
    // token at slot `row` attends to token at slot `col` within the M=4 micro-batch.
    std::array<std::array<int8_t, 4>, 4> causal_mask = {{
        {{1, 0, 0, 0}},
        {{1, 1, 0, 0}},
        {{1, 1, 1, 0}},
        {{1, 1, 1, 1}}
    }};
};

struct draft_step_candidates_m4 {
    // Depth 1 candidates sampled from root t0
    llama_token d1_top1_id   = 0;
    float       d1_top1_prob = 0.0f;
    llama_token d1_top2_id   = 0;
    float       d1_top2_prob = 0.0f;

    // Depth 2 candidates conditioned on (t0, d1_top1_id)
    llama_token d2_top1_id   = 0;
    float       d2_top1_prob = 0.0f;
    llama_token d2_top2_id   = 0;
    float       d2_top2_prob = 0.0f;

    // Depth 3 candidate conditioned on (t0, d1_top1_id, d2_top1_id)
    llama_token d3_top1_id   = 0;
    float       d3_top1_prob = 0.0f;
};

struct tree_thresholds_m4 {
    float linear_d1_min_prob  = 0.62f; // Minimum p(d1_top1) to consider depth-3 linear chain
    float linear_d2_min_prob  = 0.50f; // Minimum p(d2_top1) to sustain depth-3 linear chain
    float branch_y1_max_ratio = 2.40f; // Max ratio p(d1_top1)/p(d1_top2) to trigger root split Y1
    float branch_y1_min_second = 0.16f; // Minimum p(d1_top2) to justify allocating a slot at depth 1
    float forked_y2_min_second = 0.18f; // Minimum p(d2_top2) to justify forking at depth 2 (Y2)
};

struct kv_compaction_op_m4 {
    enum class op_type : uint8_t {
        SEQ_RM   = 0, // Remove KV entries in [p0, p1) for seq_id
        SEQ_CP   = 1, // Copy KV entries in [p0, p1) from src_seq_id to dst_seq_id
        SEQ_KEEP = 2  // Keep only dst_seq_id
    };
    op_type      type       = op_type::SEQ_RM;
    llama_seq_id src_seq_id = 0;
    llama_seq_id dst_seq_id = 0;
    llama_pos    p0         = -1;
    llama_pos    p1         = -1;
};

struct tree_verify_result_m4 {
    tree_topology_m4                topology          = tree_topology_m4::LINEAR_K3;
    int32_t                         n_draft_accepted  = 0; // In [0, 3]
    int32_t                         n_emitted_tokens  = 1; // In [1, 4] (draft accepted + 1 bonus token)
    std::array<llama_token, 4>      emitted_tokens    = {0, 0, 0, 0};
    std::array<int32_t, 4>          accepted_slots    = {0, -1, -1, -1};
    llama_seq_id                    winning_seq_id    = 0;
    llama_pos                       n_past_before     = 0;
    llama_pos                       n_past_after      = 0;
    std::vector<kv_compaction_op_m4> kv_ops           = {};
};

// Abstract KV cache mutator callback interface so the compaction algorithm can be
// bound directly to llama_kv_cache_seq_rm / cp or inspected deterministically in unit tests.
struct kv_cache_mutator_i {
    virtual ~kv_cache_mutator_i() = default;
    virtual void seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) = 0;
    virtual void seq_cp(llama_seq_id src_seq_id, llama_seq_id dst_seq_id, llama_pos p0, llama_pos p1) = 0;
};

/**
 * Selects the optimal 4-node tree topology based on draft step margin and confidence.
 *
 * Expected acceptance gain heuristics:
 * - If step 1 has high ambiguity (top-2 probability close to top-1) and a viable second branch,
 *   choose BRANCHED_Y1 to hedge root-level rejection risk.
 * - Else if step 1 is confident and step 2 is also confident, choose LINEAR_K3 to reach depth 3.
 * - Else if step 1 is confident but step 2 exhibits high entropy with a competitive runner-up,
 *   choose FORKED_Y2 to hedge depth-2 rejection risk.
 */
inline tree_topology_m4 select_tree_topology_m4(
    const draft_step_candidates_m4 & cand,
    const tree_thresholds_m4       & thresh = tree_thresholds_m4{}
) noexcept {
    const float p1_a = std::clamp(cand.d1_top1_prob, 0.0f, 1.0f);
    const float p1_b = std::clamp(cand.d1_top2_prob, 0.0f, 1.0f);
    const float p2_a = std::clamp(cand.d2_top1_prob, 0.0f, 1.0f);
    const float p2_b = std::clamp(cand.d2_top2_prob, 0.0f, 1.0f);
    const float p3_a = std::clamp(cand.d3_top1_prob, 0.0f, 1.0f);

    const bool distinct_d1 = (cand.d1_top1_id != cand.d1_top2_id);
    const bool distinct_d2 = (cand.d2_top1_id != cand.d2_top2_id);

    // Check Root-Branched Y1 trigger: step 1 has a strong runner-up and ratio is narrow
    if (distinct_d1 && p1_b >= thresh.branch_y1_min_second) {
        const float ratio_d1 = p1_a / std::max(p1_b, 1e-6f);
        const float expected_linear = p1_a * (1.0f + p2_a * (1.0f + p3_a));
        const float expected_y1     = p1_a * (1.0f + p2_a) + p1_b;
        if (ratio_d1 <= thresh.branch_y1_max_ratio || expected_y1 > expected_linear) {
            return tree_topology_m4::BRANCHED_Y1;
        }
    }

    // Check Forked Y2 trigger: step 1 is solid, step 2 has a strong runner-up that beats depth-3 linear
    if (distinct_d2 && p2_b >= thresh.forked_y2_min_second) {
        const float expected_d3_gain = p1_a * p2_a * p3_a;
        const float expected_y2_gain = p1_a * p2_b;
        if (p2_a < thresh.linear_d2_min_prob || expected_y2_gain > expected_d3_gain) {
            return tree_topology_m4::FORKED_Y2;
        }
    }

    // Default to Linear K=3 chain when step 1 and step 2 are confident or alternatives are weak
    if (p1_a >= thresh.linear_d1_min_prob && p2_a >= thresh.linear_d2_min_prob) {
        return tree_topology_m4::LINEAR_K3;
    }

    // Fallback: compare analytical expected accepted length across the 3 topologies
    const float ev_linear = p1_a * (1.0f + p2_a * (1.0f + p3_a));
    const float ev_y1     = p1_a * (1.0f + p2_a) + (distinct_d1 ? p1_b : 0.0f);
    const float ev_y2     = p1_a * (1.0f + p2_a + (distinct_d2 ? p2_b : 0.0f));

    if (ev_y1 >= ev_linear && ev_y1 >= ev_y2 && distinct_d1) {
        return tree_topology_m4::BRANCHED_Y1;
    }
    if (ev_y2 >= ev_linear && distinct_d2) {
        return tree_topology_m4::FORKED_Y2;
    }
    return tree_topology_m4::LINEAR_K3;
}

/**
 * Builds the M=4 micro-batch execution plan (tokens, relative positions, parent pointers,
 * sequence masks, and 4x4 causal mask) for the chosen topology.
 */
inline tree_batch_plan_m4 build_tree_batch_plan_m4(
    llama_token                      verified_root_token,
    const draft_step_candidates_m4 & cand,
    tree_topology_m4                 topology
) noexcept {
    tree_batch_plan_m4 plan;
    plan.topology = topology;

    switch (topology) {
        case tree_topology_m4::LINEAR_K3: {
            // Slot 0: t0 (pos +0, parent -1, seq {0})
            // Slot 1: d1_top1 (pos +1, parent 0, seq {0})
            // Slot 2: d2_top1 (pos +2, parent 1, seq {0})
            // Slot 3: d3_top1 (pos +3, parent 2, seq {0})
            plan.tokens     = {verified_root_token, cand.d1_top1_id, cand.d2_top1_id, cand.d3_top1_id};
            plan.rel_pos    = {0, 1, 2, 3};
            plan.parent_idx = {-1, 0, 1, 2};
            plan.seq_mask   = {0x01, 0x01, 0x01, 0x01};
            plan.causal_mask = {{
                {{1, 0, 0, 0}},
                {{1, 1, 0, 0}},
                {{1, 1, 1, 0}},
                {{1, 1, 1, 1}}
            }};
            break;
        }

        case tree_topology_m4::BRANCHED_Y1: {
            // Slot 0: t0      (pos +0, parent -1, shared across seq {0, 1} -> mask 0x03)
            // Slot 1: d1_top1 (pos +1, parent  0, primary branch seq {0}   -> mask 0x01)
            // Slot 2: d1_top2 (pos +1, parent  0, alt branch     seq {1}   -> mask 0x02)
            // Slot 3: d2_top1 (pos +2, parent  1, primary branch seq {0}   -> mask 0x01)
            plan.tokens     = {verified_root_token, cand.d1_top1_id, cand.d1_top2_id, cand.d2_top1_id};
            plan.rel_pos    = {0, 1, 1, 2};
            plan.parent_idx = {-1, 0, 0, 1};
            plan.seq_mask   = {0x03, 0x01, 0x02, 0x01};
            plan.causal_mask = {{
                {{1, 0, 0, 0}}, // slot 0 sees {0}
                {{1, 1, 0, 0}}, // slot 1 sees {0, 1}
                {{1, 0, 1, 0}}, // slot 2 sees {0, 2} (disjoint from slot 1)
                {{1, 1, 0, 1}}  // slot 3 sees {0, 1, 3} (disjoint from slot 2)
            }};
            break;
        }

        case tree_topology_m4::FORKED_Y2: {
            // Slot 0: t0      (pos +0, parent -1, shared across seq {0, 1} -> mask 0x03)
            // Slot 1: d1_top1 (pos +1, parent  0, shared across seq {0, 1} -> mask 0x03)
            // Slot 2: d2_top1 (pos +2, parent  1, primary branch seq {0}   -> mask 0x01)
            // Slot 3: d2_top2 (pos +2, parent  1, alt branch     seq {1}   -> mask 0x02)
            plan.tokens     = {verified_root_token, cand.d1_top1_id, cand.d2_top1_id, cand.d2_top2_id};
            plan.rel_pos    = {0, 1, 2, 2};
            plan.parent_idx = {-1, 0, 1, 1};
            plan.seq_mask   = {0x03, 0x03, 0x01, 0x02};
            plan.causal_mask = {{
                {{1, 0, 0, 0}}, // slot 0 sees {0}
                {{1, 1, 0, 0}}, // slot 1 sees {0, 1}
                {{1, 1, 1, 0}}, // slot 2 sees {0, 1, 2}
                {{1, 1, 0, 1}}  // slot 3 sees {0, 1, 3} (disjoint from slot 2)
            }};
            break;
        }
    }

    return plan;
}

/**
 * Populates a llama_batch with the M=4 tree speculation plan.
 * Strictly enforces batch.n_tokens == 4 so ggml-metal stays within the M <= 4
 * simdgroup matrix-vector hardware fast path on Apple Silicon.
 */
inline void populate_llama_batch_m4(
    llama_batch              & batch,
    const tree_batch_plan_m4 & plan,
    llama_pos                  n_past,
    llama_seq_id               base_seq_id = 0
) noexcept {
    assert(batch.token    != nullptr);
    assert(batch.pos      != nullptr);
    assert(batch.n_seq_id != nullptr);
    assert(batch.seq_id   != nullptr);
    assert(batch.logits   != nullptr);

    batch.n_tokens = kMaxSpecBatchM4;

    for (int32_t i = 0; i < kMaxSpecBatchM4; ++i) {
        batch.token[i]  = plan.tokens[i];
        batch.pos[i]    = n_past + plan.rel_pos[i];
        batch.logits[i] = 1; // Target model must emit logits for all 4 tree nodes

        int32_t seq_count = 0;
        const uint8_t mask = plan.seq_mask[i];
        if ((mask & 0x01U) != 0U) {
            batch.seq_id[i][seq_count++] = base_seq_id;
        }
        if ((mask & 0x02U) != 0U) {
            batch.seq_id[i][seq_count++] = base_seq_id + 1;
        }
        batch.n_seq_id[i] = seq_count;
    }
}

/**
 * Computes the argmax token ID over a vocabulary logit slice of length `n_vocab`.
 */
inline llama_token argmax_vocab_logits_m4(const float * logits, int32_t n_vocab) noexcept {
    assert(logits != nullptr && n_vocab > 0);
    int32_t best_id  = 0;
    float   best_val = logits[0];
    for (int32_t v = 1; v < n_vocab; ++v) {
        if (logits[v] > best_val) {
            best_val = logits[v];
            best_id  = v;
        }
    }
    return best_id;
}

/**
 * Computes temperature-scaled target probability for a specific token ID.
 */
inline float target_token_prob_m4(
    const float * logits,
    int32_t       n_vocab,
    llama_token   token_id,
    float         temperature
) noexcept {
    assert(logits != nullptr && n_vocab > 0);
    if (token_id < 0 || token_id >= n_vocab) {
        return 0.0f;
    }
    if (temperature <= 1e-5f) {
        const llama_token top_id = argmax_vocab_logits_m4(logits, n_vocab);
        return (top_id == token_id) ? 1.0f : 0.0f;
    }

    const float inv_temp = 1.0f / temperature;
    float max_logit = logits[0];
    for (int32_t v = 1; v < n_vocab; ++v) {
        if (logits[v] > max_logit) {
            max_logit = logits[v];
        }
    }

    double sum_exp = 0.0;
    for (int32_t v = 0; v < n_vocab; ++v) {
        sum_exp += std::exp(static_cast<double>(logits[v] - max_logit) * static_cast<double>(inv_temp));
    }
    const double tok_exp = std::exp(static_cast<double>(logits[token_id] - max_logit) * static_cast<double>(inv_temp));
    return static_cast<float>(tok_exp / std::max(sum_exp, 1e-20));
}

/**
 * Verifies the 4-node speculative tree against the target model's output logits
 * (`target_logits` of shape [4, n_vocab]) and executes exact multi-branch KV-cache
 * compaction so that `base_seq_id` holds only the accepted prefix positions
 * `[0, n_past_after)` and `base_seq_id + 1` is completely cleared.
 *
 * Supports both greedy verification (`temperature <= 0.0f`) and stochastic
 * speculative rejection sampling (`u_rand[step] < min(1, p_target / p_draft)`).
 */
inline tree_verify_result_m4 verify_tree_and_compact_kv_m4(
    const tree_batch_plan_m4       & plan,
    const draft_step_candidates_m4 & cand,
    const float                    * target_logits,
    int32_t                          n_vocab,
    llama_pos                        n_past,
    llama_seq_id                     base_seq_id    = 0,
    float                            temperature    = 0.0f,
    const std::array<float, 3>     & u_rand         = {0.0f, 0.0f, 0.0f},
    kv_cache_mutator_i             * kv_mutator     = nullptr
) {
    assert(target_logits != nullptr && n_vocab > 0);

    tree_verify_result_m4 res;
    res.topology       = plan.topology;
    res.n_past_before  = n_past;
    res.winning_seq_id = base_seq_id;
    res.accepted_slots = {0, -1, -1, -1};

    auto slot_logits = [&](int32_t slot) -> const float * {
        return target_logits + static_cast<std::size_t>(slot) * static_cast<std::size_t>(n_vocab);
    };

    auto accept_candidate = [&](int32_t parent_slot, llama_token draft_tok, float draft_prob, float u_val) -> bool {
        const float * l_ptr = slot_logits(parent_slot);
        if (temperature <= 1e-5f) {
            return argmax_vocab_logits_m4(l_ptr, n_vocab) == draft_tok;
        }
        const float p_target = target_token_prob_m4(l_ptr, n_vocab, draft_tok, temperature);
        const float p_draft  = std::max(draft_prob, 1e-6f);
        const float ratio    = std::min(1.0f, p_target / p_draft);
        return u_val < ratio;
    };

    switch (plan.topology) {
        case tree_topology_m4::LINEAR_K3: {
            // Path: 0 -> 1 -> 2 -> 3
            int32_t terminal_slot = 0;
            if (accept_candidate(0, plan.tokens[1], cand.d1_top1_prob, u_rand[0])) {
                res.emitted_tokens[res.n_draft_accepted++] = plan.tokens[1];
                res.accepted_slots[1] = 1;
                terminal_slot = 1;

                if (accept_candidate(1, plan.tokens[2], cand.d2_top1_prob, u_rand[1])) {
                    res.emitted_tokens[res.n_draft_accepted++] = plan.tokens[2];
                    res.accepted_slots[2] = 2;
                    terminal_slot = 2;

                    if (accept_candidate(2, plan.tokens[3], cand.d3_top1_prob, u_rand[2])) {
                        res.emitted_tokens[res.n_draft_accepted++] = plan.tokens[3];
                        res.accepted_slots[3] = 3;
                        terminal_slot = 3;
                    }
                }
            }

            // Sample/argmax bonus token from the deepest accepted node's target logits
            const llama_token bonus_tok = argmax_vocab_logits_m4(slot_logits(terminal_slot), n_vocab);
            res.emitted_tokens[res.n_draft_accepted] = bonus_tok;
            res.n_emitted_tokens = res.n_draft_accepted + 1;
            res.n_past_after     = n_past + res.n_emitted_tokens;

            // Evict rejected speculative KV entries from [n_past_after, n_past + 4) on base_seq_id
            if (res.n_past_after < n_past + kMaxSpecBatchM4) {
                res.kv_ops.push_back({
                    kv_compaction_op_m4::op_type::SEQ_RM,
                    base_seq_id,
                    base_seq_id,
                    res.n_past_after,
                    n_past + kMaxSpecBatchM4
                });
            }
            break;
        }

        case tree_topology_m4::BRANCHED_Y1: {
            // Slot 0: root (pos +0, seq 0 & 1)
            // Primary branch (seq 0): Slot 1 (pos +1) -> Slot 3 (pos +2)
            // Secondary branch (seq 1): Slot 2 (pos +1)
            int32_t terminal_slot = 0;
            bool    used_alt_seq  = false;

            if (accept_candidate(0, plan.tokens[1], cand.d1_top1_prob, u_rand[0])) {
                // Primary branch accepted at depth 1
                res.emitted_tokens[res.n_draft_accepted++] = plan.tokens[1];
                res.accepted_slots[1] = 1;
                terminal_slot = 1;

                if (accept_candidate(1, plan.tokens[3], cand.d2_top1_prob, u_rand[1])) {
                    res.emitted_tokens[res.n_draft_accepted++] = plan.tokens[3];
                    res.accepted_slots[2] = 3;
                    terminal_slot = 3;
                }
            } else if (accept_candidate(0, plan.tokens[2], cand.d1_top2_prob, u_rand[0])) {
                // Secondary branch accepted at depth 1 (slot 2 on base_seq_id + 1)
                res.emitted_tokens[res.n_draft_accepted++] = plan.tokens[2];
                res.accepted_slots[1] = 2;
                terminal_slot = 2;
                used_alt_seq  = true;
            }

            const llama_token bonus_tok = argmax_vocab_logits_m4(slot_logits(terminal_slot), n_vocab);
            res.emitted_tokens[res.n_draft_accepted] = bonus_tok;
            res.n_emitted_tokens = res.n_draft_accepted + 1;
            res.n_past_after     = n_past + res.n_emitted_tokens;
            res.winning_seq_id   = used_alt_seq ? (base_seq_id + 1) : base_seq_id;

            if (used_alt_seq) {
                // 1. Remove rejected primary branch positions [n_past + 1, n_past + 3) on base_seq_id
                res.kv_ops.push_back({
                    kv_compaction_op_m4::op_type::SEQ_RM,
                    base_seq_id,
                    base_seq_id,
                    n_past + 1,
                    n_past + 3
                });
                // 2. Copy accepted slot 2 at position [n_past + 1, n_past + 2) from alt seq to base_seq_id
                res.kv_ops.push_back({
                    kv_compaction_op_m4::op_type::SEQ_CP,
                    base_seq_id + 1,
                    base_seq_id,
                    n_past + 1,
                    n_past + 2
                });
            } else {
                // Primary branch won (or both rejected at depth 1): trim unaccepted tail on base_seq_id
                if (res.n_past_after < n_past + 3) {
                    res.kv_ops.push_back({
                        kv_compaction_op_m4::op_type::SEQ_RM,
                        base_seq_id,
                        base_seq_id,
                        res.n_past_after,
                        n_past + 3
                    });
                }
            }

            // Always purge auxiliary sequence (base_seq_id + 1) speculative positions [n_past, n_past + 3)
            res.kv_ops.push_back({
                kv_compaction_op_m4::op_type::SEQ_RM,
                base_seq_id + 1,
                base_seq_id + 1,
                n_past,
                n_past + 3
            });
            break;
        }

        case tree_topology_m4::FORKED_Y2: {
            // Slot 0: root (pos +0, seq 0 & 1)
            // Slot 1: d1_top1 (pos +1, seq 0 & 1)
            // Primary fork (seq 0): Slot 2 (pos +2)
            // Secondary fork (seq 1): Slot 3 (pos +2)
            int32_t terminal_slot = 0;
            bool    used_alt_seq  = false;

            if (accept_candidate(0, plan.tokens[1], cand.d1_top1_prob, u_rand[0])) {
                res.emitted_tokens[res.n_draft_accepted++] = plan.tokens[1];
                res.accepted_slots[1] = 1;
                terminal_slot = 1;

                if (accept_candidate(1, plan.tokens[2], cand.d2_top1_prob, u_rand[1])) {
                    res.emitted_tokens[res.n_draft_accepted++] = plan.tokens[2];
                    res.accepted_slots[2] = 2;
                    terminal_slot = 2;
                } else if (accept_candidate(1, plan.tokens[3], cand.d2_top2_prob, u_rand[1])) {
                    res.emitted_tokens[res.n_draft_accepted++] = plan.tokens[3];
                    res.accepted_slots[2] = 3;
                    terminal_slot = 3;
                    used_alt_seq  = true;
                }
            }

            const llama_token bonus_tok = argmax_vocab_logits_m4(slot_logits(terminal_slot), n_vocab);
            res.emitted_tokens[res.n_draft_accepted] = bonus_tok;
            res.n_emitted_tokens = res.n_draft_accepted + 1;
            res.n_past_after     = n_past + res.n_emitted_tokens;
            res.winning_seq_id   = used_alt_seq ? (base_seq_id + 1) : base_seq_id;

            if (used_alt_seq) {
                // Evict primary fork slot 2 at [n_past + 2, n_past + 3) on base_seq_id
                res.kv_ops.push_back({
                    kv_compaction_op_m4::op_type::SEQ_RM,
                    base_seq_id,
                    base_seq_id,
                    n_past + 2,
                    n_past + 3
                });
                // Copy secondary fork slot 3 at [n_past + 2, n_past + 3) from base_seq_id + 1 to base_seq_id
                res.kv_ops.push_back({
                    kv_compaction_op_m4::op_type::SEQ_CP,
                    base_seq_id + 1,
                    base_seq_id,
                    n_past + 2,
                    n_past + 3
                });
            } else {
                if (res.n_past_after < n_past + 3) {
                    res.kv_ops.push_back({
                        kv_compaction_op_m4::op_type::SEQ_RM,
                        base_seq_id,
                        base_seq_id,
                        res.n_past_after,
                        n_past + 3
                    });
                }
            }

            // Purge auxiliary sequence (base_seq_id + 1) across [n_past, n_past + 3)
            res.kv_ops.push_back({
                kv_compaction_op_m4::op_type::SEQ_RM,
                base_seq_id + 1,
                base_seq_id + 1,
                n_past,
                n_past + 3
            });
            break;
        }
    }

    if (kv_mutator != nullptr) {
        for (const auto & op : res.kv_ops) {
            if (op.type == kv_compaction_op_m4::op_type::SEQ_RM) {
                kv_mutator->seq_rm(op.src_seq_id, op.p0, op.p1);
            } else if (op.type == kv_compaction_op_m4::op_type::SEQ_CP) {
                kv_mutator->seq_cp(op.src_seq_id, op.dst_seq_id, op.p0, op.p1);
            }
        }
    }

    return res;
}

} // namespace prism::m4
