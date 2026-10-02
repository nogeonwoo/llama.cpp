#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "log.h"
#include "llama.h"
#include "speculative-async.h"
#include "../src/llama-ext.h"
#include "speculative-tap.h"

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <cstring>
#include <cinttypes>
#include <string>
#include <vector>
#include <utility>
#include <fstream>

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    if (params.n_predict < -1) {
        LOG_ERR("%s: --n-predict must be >= -1\n", __func__);
        return 1;
    }

    const auto output_limits = common_speculative_get_output_limits(
            params.n_batch, params.n_parallel, common_speculative_n_max(&params.speculative));
    params.n_outputs_max = output_limits.total;
    params.n_outputs_max_per_seq = output_limits.per_seq;

    // Stage S8: Install Tap Feature callback into target backend scheduler
    static spec::layer_tap g_tap;
    const char * env_tap_verify = std::getenv("LLAMA_SPEC_TAP_VERIFY");
    const bool enable_tap_gate = env_tap_verify ? (std::atoi(env_tap_verify) != 0) : true;
    if (enable_tap_gate) {
        params.cb_eval = spec::layer_tap::eval_cb;
        params.cb_eval_user_data = &g_tap;
    }

    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model_tgt = NULL;

    llama_context * ctx_tgt = NULL;

    // load the target model
    auto llama_init_tgt = common_init_from_params(params);

    model_tgt = llama_init_tgt->model();
    ctx_tgt   = llama_init_tgt->context();

    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);

    // load the draft model (if any) - this also creates the MTP draft context when MTP speculation is enabled
    common_speculative_init_result_ptr spec_init;

    {
        common_params params_dft = common_base_params_to_speculative(params);

        spec_init = common_speculative_init_from_params(params_dft, model_tgt, ctx_tgt);

        params.speculative.draft.ctx_tgt = ctx_tgt;
        params.speculative.draft.ctx_dft = spec_init->context();
    }

    llama_context * ctx_dft = params.speculative.draft.ctx_dft;

    if (enable_tap_gate && ctx_dft) {
        const llama_model * model_dft = llama_get_model(ctx_dft);
        const int32_t * target_layer_ids = llama_model_target_layer_ids(model_dft);
        const uint32_t target_layer_ids_n = llama_model_target_layer_ids_n(model_dft);
        const int64_t n_embd_tgt = llama_model_n_embd(model_tgt);
        g_tap.configure(target_layer_ids, target_layer_ids_n, n_embd_tgt);
        std::string s_layers;
        for (uint32_t i = 0; i < target_layer_ids_n; ++i) {
            s_layers += std::to_string(target_layer_ids[i]) + (i + 1 < target_layer_ids_n ? ", " : "");
        }
        LOG_INF("[tap-gate] Configured layer tap for %u target layers (layers=[%s], n_embd=%" PRId64 ")\n",
                target_layer_ids_n, s_layers.c_str(), n_embd_tgt);
    }

    // check if the context supports partial sequence removal
    const common_context_seq_rm_type seq_rm_tgt = common_context_can_seq_rm(ctx_tgt);
    const common_context_seq_rm_type seq_rm_dft = ctx_dft ? common_context_can_seq_rm(ctx_dft) : COMMON_CONTEXT_SEQ_RM_TYPE_NO;

    // a bounded rollback window (RS) can only undo up to llama_n_rs_seq() tokens, so longer drafts still take a checkpoint (same rule as the server)
    bool use_ckpt_tgt = seq_rm_tgt == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
    bool use_ckpt_dft = seq_rm_dft == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

    if (use_ckpt_tgt) {
        LOG_INF("speculative decoding will use checkpoints (context does not support partial sequence removal)\n");
    }

    // Tokenize the prompt
    std::vector<llama_token> inp;
    inp = common_tokenize(ctx_tgt, params.prompt, true, true);

    if (llama_n_ctx(ctx_tgt) < (uint32_t) inp.size()) {
        LOG_ERR("%s: the prompt exceeds the context size (%d tokens, ctx %d)\n", __func__, (int) inp.size(), llama_n_ctx(ctx_tgt));

        return 1;
    }

    if (llama_n_batch(ctx_tgt) < (uint32_t) inp.size()) {
        LOG_ERR("%s: the prompt exceeds the batch size (%d tokens, batch %d)\n", __func__, (int) inp.size(), llama_n_batch(ctx_tgt));

        return 1;
    }

    LOG("\n\n");

    for (auto id : inp) {
        LOG("%s", common_token_to_piece(ctx_tgt, id).c_str());
    }

    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;

    // used to determine end of generation
    bool has_eos = false;

    llama_seq_id seq_id = 0;

    // ================================================
    // everything until here is standard initialization
    // the relevant stuff for speculative decoding starts here

    const auto t_enc_start = ggml_time_us();

    // target model sampling context
    common_sampler_ptr smpl(common_sampler_init(model_tgt, params.sampling));

    // init the speculator
    const auto & params_spec = params.speculative;

    struct common_speculative * spec = common_speculative_init(params.speculative, 1);

    if (spec == nullptr) {
        LOG_ERR("%s", "failed to initialize speculative decoding\n");
        return 1;
    }

    // eval the prompt on the target and feed it to the speculative implementation(s)
    {
        llama_batch batch_prompt = llama_batch_init(inp.size(), 0, 1);
        for (size_t i = 0; i < inp.size() - 1; ++i) {
            common_batch_add(batch_prompt, inp[i], i, { seq_id }, false);
        }

        llama_decode(ctx_tgt, batch_prompt);

        if (!common_speculative_process(spec, batch_prompt)) {
            LOG_ERR("%s", "failed to process speculative prompt\n");
            return 1;
        }
    }

    // note: keep the last token separate!
    llama_token id_last = inp.back();

    // all tokens currently in the target context
    llama_tokens prompt_tgt(inp.begin(), inp.end() - 1);
    prompt_tgt.reserve(llama_n_ctx(ctx_tgt));

    int n_past = inp.size() - 1;

    common_speculative_begin(spec, seq_id, prompt_tgt);

    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);

    llama_tokens draft;

    common_prompt_checkpoint ckpt;

    const auto t_enc_end = ggml_time_us();

    struct cycle_profile {
        int cycle_idx;
        int k_drafted;
        int64_t t_draft_us;
        int64_t t_target_us;
        int64_t t_process_us;
        int64_t t_sample_us;
        int64_t t_rollback_us;
        int64_t t_host_us;
        int64_t t_cycle_us;
        int n_drafted;
        int n_accepted;
        int n_generated;
        std::vector<int> accepted_positions;
        std::vector<double> c;
        std::vector<double> a;
    };
    std::vector<cycle_profile> cycle_profiles;
    int cycle_idx = 0;

    // Stage S8: Asynchronous Pipelining state and telemetry
    const char * env_pipe = std::getenv("LLAMA_SPEC_PIPELINE");
    const bool opt_pipeline = env_pipe && (std::strcmp(env_pipe, "1") == 0 || std::strcmp(env_pipe, "true") == 0);

    // Worker pipeline: needs the tap callback on the target context (enable_tap_gate), n_seq == 1 and the DFlash impl.
    const bool pipe_on = opt_pipeline && enable_tap_gate && common_speculative_pipeline_attach(spec, &g_tap);
    if (opt_pipeline && !pipe_on) {
        LOG_WRN("[spec-pipeline] LLAMA_SPEC_PIPELINE=1 but the worker pipeline could not be attached "
                "(needs LLAMA_SPEC_TAP_VERIFY!=0, one sequence, a DFlash drafter and a tail-droppable draft KV cache); running sequentially\n");
    }

    // Stage S8 cycle decomposition accumulators
    int64_t sum_t_target_enq_us  = 0;
    int64_t sum_t_target_sync_us = 0;
    int64_t sum_t_draft_us       = 0;
    int64_t sum_t_process_us     = 0;
    int64_t sum_t_sample_us      = 0;
    int64_t sum_t_cycle_us       = 0;
    int64_t total_cycles_count   = 0;

    const auto t_dec_start = ggml_time_us();

    while (true) {
        const int64_t t_cyc_start = ggml_time_us();
        int64_t t_draft_us = 0;
        int64_t t_target_us = 0;
        int64_t t_target_enq_us = 0;
        int64_t t_target_sync_us = 0;

        // generate or reuse draft tokens
        //
        // this is the most important part of the speculation. the more probable tokens that are provided here
        // the better the performance will be. in theory, this computation can be performed asynchronously and even
        // offloaded to a remote device. it doesn't even have to be based on an LLM. instead, it can provide tokens
        // from a cache or lookup tables.
        //
        if (draft.empty()) {
            ckpt.update_pos(
                    prompt_tgt.size(),
                    llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), seq_id),
                    llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), seq_id));

            if (seq_rm_dft == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
                ckpt.update_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }

            // determine the max draft that fits the remaining context and generation budget
            int n_draft_max = (int) llama_n_ctx(ctx_tgt) - n_past - 2;
            if (params.n_predict >= 0) {
                n_draft_max = std::min(n_draft_max, params.n_predict - n_predict - 1);
            }
            n_draft_max = std::max(n_draft_max, 0);

            // generate a new draft: the worker already drafted it behind the previous verify when the hypothesis hit
            // and the arbiter still wants the same K, otherwise draft sequentially (the draft KV cache is already
            // ingested either way: the worker did it from the tap, or process() did it)
            const int64_t t_d_start = ggml_time_us();
            const bool drafted_by_worker = pipe_on && common_speculative_pipeline_take(spec, seq_id, n_draft_max, draft);
            if (!drafted_by_worker) {
                common_speculative_get_draft_params(spec, seq_id) = {
                    /* .drafting   = */ true,
                    /* .n_max      = */ n_draft_max,
                    /* .n_past     = */ n_past,
                    /* .id_last    = */ id_last,
                    /* .prompt     = */ &prompt_tgt,
                    /* .result     = */ &draft, // output
                };
                common_speculative_draft(spec);
            }
            t_draft_us = ggml_time_us() - t_d_start;

            // reset the draft context to the checkpoint before verification
            if (ctx_dft) {
                // only a FULL context was checkpointed above; an RS context is handled after the draft is sized
                if (seq_rm_dft == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
                    ckpt.load_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }

                llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, ckpt.pos_max + 1, -1);
            }

            // save a checkpoint of the target context before evaluating the draft
            // this allows us to restore the state if partial draft acceptance occurs
            if (!draft.empty()) {
                use_ckpt_tgt = seq_rm_tgt == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                              (seq_rm_tgt == COMMON_CONTEXT_SEQ_RM_TYPE_RS && draft.size() > llama_n_rs_seq(ctx_tgt));
                const bool ckpt_dft_rs = seq_rm_dft == COMMON_CONTEXT_SEQ_RM_TYPE_RS && draft.size() > llama_n_rs_seq(ctx_dft);
                use_ckpt_dft = seq_rm_dft == COMMON_CONTEXT_SEQ_RM_TYPE_FULL || ckpt_dft_rs;

                if (use_ckpt_tgt) {
                    ckpt.update_tgt(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }

                // a FULL draft context was already checkpointed before drafting; only the RS overflow case needs one here
                if (ckpt_dft_rs) {
                    ckpt.update_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }
            }
        } else {
            // we have a previous (partial) draft to reuse from checkpoint restoration
            if (use_ckpt_tgt) {
                GGML_ASSERT(!ckpt.empty());
            }
        }

        // always have a token to evaluate from before - id_last
        common_batch_clear(batch_tgt);
        common_batch_add  (batch_tgt, id_last, n_past++, { seq_id }, true);

        // true when the worker pipeline owns the tap for this verify batch (it then also verifies the features)
        bool pipe_armed = false;

        // evaluate the target model on [id_last, draft0, draft1, ..., draftN-1]
        {
            for (size_t i = 0; i < draft.size(); ++i) {
                common_batch_add(batch_tgt, draft[i], n_past + i, { seq_id }, true);
            }

            // masked-4: M=3 (280.1 ms) -> M=4 (221.5 ms); dummy row: valid token, contiguous pos, same seq, no logits
            const bool      pad4    = (draft.size() == 2);
            const llama_pos pos_pad = n_past + (llama_pos) draft.size();
            if (pad4) {
                common_batch_add(batch_tgt, draft.back(), pos_pad, { seq_id }, false);
            }

            const int32_t n_real = (int32_t)draft.size() + 1;
            pipe_armed = pipe_on && common_speculative_pipeline_arm(spec, seq_id, batch_tgt.pos[0], n_real, (int32_t) draft.size());
            if (!pipe_armed && enable_tap_gate && g_tap.configured()) {
                g_tap.arm(n_real, nullptr);
            }

            const int64_t t_target_start = ggml_time_us();
            llama_decode(ctx_tgt, batch_tgt);
            t_target_enq_us = ggml_time_us() - t_target_start;

            const int64_t t_sync_start = ggml_time_us();
            llama_synchronize(ctx_tgt);
            t_target_sync_us = ggml_time_us() - t_sync_start;
            t_target_us = ggml_time_us() - t_target_start;

            if (pad4) {
                batch_tgt.n_tokens--;
            }

            if (pipe_armed) {
                common_speculative_pipeline_disarm(spec, t_target_start + t_target_us);
            } else if (enable_tap_gate && g_tap.configured()) {
                g_tap.disarm();
            }
        }

        // Defensive invariant (ADR 0008): dummy tokens from M=4 padding must never leak to drafter
        GGML_ASSERT(batch_tgt.n_tokens == (int32_t)draft.size() + 1);

        // -----------------------------------------------------------------------------------------
        // Stage S8 Correctness Gate: Tap Feature Bit-Exact Verification
        // -----------------------------------------------------------------------------------------
        if (!pipe_armed && enable_tap_gate && g_tap.configured()) {
            const int32_t n_real = (int32_t)draft.size() + 1;
            if (!g_tap.captured()) {
                LOG_ERR("[tap-gate cyc=%d K=%zu] FAILED: layer tap did not capture features (stats: armed=%" PRIu64 ", fired=%" PRIu64 ", failed=%" PRIu64 ")\n",
                        cycle_idx, draft.size(), g_tap.stats().armed, g_tap.stats().fired, g_tap.stats().failed);
                GGML_ASSERT(false && "Tap Correctness Gate FAILED: Tap did not fire or name mismatch!");
            } else {
                const int64_t n_embd = llama_model_n_embd(model_tgt);
                bool cycle_bit_exact = true;

                for (size_t s = 0; s < g_tap.n_slots(); ++s) {
                    const uint32_t lid = g_tap.slot_layer(s);
                    const float * tapped = g_tap.rows(s);
                    const float * direct = llama_get_embeddings_layer_inp(ctx_tgt, lid);

                    if (!direct) {
                        LOG_ERR("[tap-gate cyc=%d K=%zu slot=%zu lid=%u] FAILED: llama_get_embeddings_layer_inp returned NULL!\n",
                                cycle_idx, draft.size(), s, lid);
                        GGML_ASSERT(false && "Tap Correctness Gate FAILED: Target layer input is NULL!");
                    }

                    for (int32_t r = 0; r < n_real; ++r) {
                        const float * row_tap = tapped + (size_t) r * n_embd;
                        const float * row_dir = direct + (size_t) r * n_embd;
                        if (std::memcmp(row_tap, row_dir, (size_t) n_embd * sizeof(float)) != 0) {
                            cycle_bit_exact = false;
                            size_t first_mismatch_idx = 0;
                            float val_tapped = 0.0f;
                            float val_direct = 0.0f;
                            for (int64_t e = 0; e < n_embd; ++e) {
                                if (row_tap[e] != row_dir[e]) {
                                    first_mismatch_idx = (size_t) r * n_embd + e;
                                    val_tapped = row_tap[e];
                                    val_direct = row_dir[e];
                                    break;
                                }
                            }
                            uint32_t hex_tapped = 0, hex_direct = 0;
                            std::memcpy(&hex_tapped, &val_tapped, sizeof(float));
                            std::memcpy(&hex_direct, &val_direct, sizeof(float));
                            LOG_ERR("[tap-gate cyc=%d K=%zu slot=%zu lid=%u row=%d] MISMATCH at flat_idx=%zu: tapped=%.7f (0x%08X) vs direct=%.7f (0x%08X)!\n",
                                    cycle_idx, draft.size(), s, lid, r, first_mismatch_idx,
                                    val_tapped, hex_tapped,
                                    val_direct, hex_direct);
                            GGML_ASSERT(false && "Tap Bit-Exact Correctness Gate FAILED: Tapped features differ from Target features!");
                        }
                    }
                }

                if (cycle_bit_exact) {
                    LOG_INF("[tap-gate cyc=%d K=%zu] BIT-EXACT MATCH: all %zu layers x %d rows (%" PRIu64 " float32 elements 100%% bitwise identical)\n",
                            cycle_idx, draft.size(), g_tap.n_slots(), n_real, (uint64_t) g_tap.n_slots() * (uint64_t) n_real * (uint64_t) n_embd);
                }
            }
        }

        // feed the batch to the speculative implementation(s) - this drives the draft model, MTP, Eagle3, etc.
        const int64_t t_proc_start = ggml_time_us();
        if (!common_speculative_process(spec, batch_tgt)) {
            LOG_ERR("%s", "failed to process speculative batch\n");
            break;
        }
        const int64_t t_process_us = ggml_time_us() - t_proc_start;

        // only save the sampler sampler state if we use checkpoints
        common_sampler_ptr smpl_save;
        if (use_ckpt_tgt) {
            smpl_save.reset(common_sampler_clone(smpl.get()));
        }

        // save the size of the draft being verified
        const size_t n_draft = draft.size();

        // sample from the full target batch and return the accepted tokens based on the target sampler
        //
        // for each token to be accepted, the sampler would have to sample that same token
        // in such cases, instead of decoding the sampled token as we normally do, we simply continue with the
        // available logits from the batch and sample the next token until we run out of logits or the sampler
        // disagrees with the draft
        //
        const int64_t t_sample_start = ggml_time_us();
        auto ids = common_sampler_sample_and_accept_n(smpl.get(), ctx_tgt, draft);

        //LOG_DBG("ids: %s\n", string_from(ctx_tgt, ids).c_str());

        GGML_ASSERT(ids.size() > 0); // there will always be at least one accepted token

        // join the worker (ctx_dft is the main thread's again after this), verify the tapped features, decide hit/miss
        if (pipe_on) {
            common_speculative_pipeline_collect(spec, seq_id, (int) ids.size() - 1, ids.back());
        }

        // check for partial draft acceptance:
        // if the context doesn't support partial sequence removal, restore the checkpoint
        // and make the accepted tokens the new partial draft for the next iteration
        if (use_ckpt_tgt && ids.size() - 1 < n_draft) {
            LOG_DBG("partial acceptance: %zu < %zu, restoring checkpoint\n", ids.size() - 1, n_draft);

            if (pipe_on) {
                common_speculative_pipeline_abort(spec);
            }

            draft = std::move(ids);

            {
                ckpt.load_tgt(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq_id, ckpt.pos_max + 1, -1);
            }

            if (ctx_dft) {
                if (use_ckpt_dft) {
                    ckpt.load_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }

                llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, ckpt.pos_max + 1, -1);
            }

            prompt_tgt.resize(ckpt.n_tokens);
            smpl = std::move(smpl_save);

            n_past = (int) prompt_tgt.size();

            continue;
        }

        common_speculative_accept(spec, seq_id, ids.size() - 1);

        // full acceptance: consume the draft and commit accepted tokens
        n_past    += ids.size() - 1;
        n_drafted += n_draft; // note: we ignore the discarded small drafts
        n_accept  += ids.size() - 1;
        n_predict += ids.size();

        // process the accepted tokens and update contexts
        //
        // this is the standard token post-processing that we normally do
        // in this case, we do it for a group of accepted tokens at once
        //
        for (size_t i = 0; i < ids.size(); ++i) {
            prompt_tgt.push_back(id_last);

            id_last = ids[i];

            if (llama_vocab_is_eog(vocab, id_last)) {
                has_eos = true;
                break;
            }

            const std::string token_str = common_token_to_piece(ctx_tgt, id_last);

            if (params.use_color && i + 1 < ids.size()) {
                LOG("\u001b[%dm%s\u001b[37m", (36 - 0 % 6), token_str.c_str());
            } else {
                LOG("%s", token_str.c_str());
            }
        }
        const int64_t t_sample_us = ggml_time_us() - t_sample_start;

        LOG_DBG("accepted %d/%d draft tokens, the last target token is: (%d)\n", (int) ids.size() - 1, (int) draft.size(), id_last);

        // clear the draft since it has been consumed
        draft.clear();

        const int64_t t_rollback_start = ggml_time_us();
        {
            LOG_DBG("clear kv cache from any extra tokens, n_past = %d\n", n_past);

            llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq_id, n_past, -1);

            if (ctx_dft) {
                llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, n_past, -1);
            }
        }
        const int64_t t_rollback_us = ggml_time_us() - t_rollback_start;

        const int64_t t_cyc_end = ggml_time_us();
        const int64_t t_cycle_us = t_cyc_end - t_cyc_start;
        const int64_t t_host_us = t_cycle_us - t_target_us - t_draft_us;

        sum_t_target_enq_us  += t_target_enq_us;
        sum_t_target_sync_us += t_target_sync_us;
        sum_t_draft_us       += t_draft_us;
        sum_t_process_us     += t_process_us;
        sum_t_sample_us      += t_sample_us;
        sum_t_cycle_us       += t_cycle_us;
        total_cycles_count++;

        const int n_acc = (int) ids.size() - 1;
        const int n_drafted_cur = (int) n_draft;
        const int n_gen_cur = (int) ids.size();

        std::vector<int> acc_pos;
        std::vector<double> c_pos;
        std::vector<double> a_pos;
        for (int p = 1; p <= n_drafted_cur; ++p) {
            const int acc = (n_acc >= p) ? 1 : 0;
            acc_pos.push_back(acc);
            a_pos.push_back((double) acc);
            if (p == 1) {
                c_pos.push_back((double) acc);
            } else {
                c_pos.push_back((n_acc >= p - 1) ? (double) acc : 0.0);
            }
        }

        cycle_profiles.push_back({
            cycle_idx++,
            n_drafted_cur,
            t_draft_us,
            t_target_us,
            t_process_us,
            t_sample_us,
            t_rollback_us,
            t_host_us,
            t_cycle_us,
            n_drafted_cur,
            n_acc,
            n_gen_cur,
            acc_pos,
            c_pos,
            a_pos
        });

        if ((params.n_predict >= 0 && n_predict > params.n_predict) || has_eos) {
            break;
        }
    }

    auto t_dec_end = ggml_time_us();

    const int n_input = inp.size();

    LOG("\n\n");

    LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict  / ((t_dec_end - t_dec_start) / 1e6f));

    LOG_INF("\n");
    LOG_INF("n_draft   = %d\n", params_spec.draft.n_max);
    LOG_INF("n_predict = %d\n", n_predict);
    LOG_INF("n_drafted = %d\n", n_drafted);
    LOG_INF("n_accept  = %d\n", n_accept);
    LOG_INF("accept    = %.3f%%\n", 100.0f * n_accept / n_drafted);

    if (opt_pipeline) {
        const spec::pipe_stats ps = common_speculative_pipeline_get_stats(spec);
        const double n_sub = (double) std::max<uint64_t>(ps.submitted, 1);
        LOG_INF("\n");
        LOG_INF("[spec-pipeline] Stage S8 Asynchronous Pipelining Telemetry (attached=%d, disabled=%d):\n", (int) pipe_on, (int) ps.disabled);
        LOG_INF("[spec-pipeline] - Hypotheses submitted : %" PRIu64 "\n", ps.submitted);
        LOG_INF("[spec-pipeline] - Ingest-only jobs done : %" PRIu64 "\n", ps.ingest_jobs);
        LOG_INF("[spec-pipeline] - Hits (draft consumed): %" PRIu64 " (%.2f%% of submitted)\n", ps.hits, 100.0 * (double) ps.hits / n_sub);
        LOG_INF("[spec-pipeline] - Misses: accept=%" PRIu64 " anchor=%" PRIu64 " k=%" PRIu64 " empty=%" PRIu64 "\n",
                ps.miss_accept, ps.miss_anchor, ps.miss_k, ps.miss_empty);
        LOG_INF("[spec-pipeline] - Gate failures: verify=%" PRIu64 " tap=%" PRIu64 "\n", ps.verify_fail, ps.tap_fail);
        LOG_INF("[spec-pipeline] - Sequential drafts: lookahead on=%" PRIu64 " off=%" PRIu64 "\n", ps.la_on_seq, ps.la_off_seq);
        LOG_INF("[spec-pipeline] - Per submitted job (ms): worker=%.2f wait=%.2f window=%.2f tap_cb=%.2f\n",
                ps.t_worker_us / 1000.0 / n_sub, ps.t_wait_us / 1000.0 / n_sub,
                ps.t_window_us / 1000.0 / n_sub, ps.t_tap_cb_us / 1000.0 / n_sub);
    }

    LOG_INF("\n================================================================================\n");
    LOG_INF("           STAGE S8 CYCLE LATENCY DECOMPOSITION BREAKDOWN (M3 EMPIRICAL)        \n");
    LOG_INF("================================================================================\n");
    if (total_cycles_count > 0 && sum_t_cycle_us > 0) {
        LOG_INF("  Total Speculative Cycles      : %" PRId64 "\n", total_cycles_count);
        LOG_INF("  Avg Target Decode CPU Enqueue : %8.2f ms\n", (sum_t_target_enq_us  / (double) total_cycles_count) / 1000.0);
        LOG_INF("  Avg Target Metal GPU Wait     : %8.2f ms\n", (sum_t_target_sync_us / (double) total_cycles_count) / 1000.0);
        LOG_INF("  Avg Total Target Verify (tv)  : %8.2f ms\n", ((sum_t_target_enq_us + sum_t_target_sync_us) / (double) total_cycles_count) / 1000.0);
        LOG_INF("  Avg DFlash Draft Pass (td)    : %8.2f ms\n", (sum_t_draft_us       / (double) total_cycles_count) / 1000.0);
        LOG_INF("  Avg DFlash Feature Ingest     : %8.2f ms\n", (sum_t_process_us     / (double) total_cycles_count) / 1000.0);
        LOG_INF("  Avg Sampler + Accept          : %8.2f ms\n", (sum_t_sample_us      / (double) total_cycles_count) / 1000.0);
        LOG_INF("  Avg End-to-End Cycle Time (C) : %8.2f ms\n", (sum_t_cycle_us       / (double) total_cycles_count) / 1000.0);
        LOG_INF("--------------------------------------------------------------------------------\n");
        LOG_INF("  Draft Latency Fraction (td/C) : %8.2f %%\n", 100.0 * (sum_t_draft_us / (double) sum_t_cycle_us));
    }
    LOG_INF("================================================================================\n");

    LOG_INF("\n");
    LOG_INF("draft:\n\n");
    common_speculative_print_stats(spec);

    LOG_INF("\n");
    LOG_INF("target:\n\n");
    common_perf_print(ctx_tgt, smpl.get());

    const char * env_cycle_log = std::getenv("LLAMA_SPEC_CYCLE_LOG");
    if (env_cycle_log && env_cycle_log[0] != '\0') {
        std::ofstream ofs(env_cycle_log);
        if (ofs.is_open()) {
            ofs << "[\n";
            for (size_t i = 0; i < cycle_profiles.size(); ++i) {
                const auto & cp = cycle_profiles[i];
                ofs << "  {\n";
                ofs << "    \"cycle_idx\": " << cp.cycle_idx << ",\n";
                ofs << "    \"k_drafted\": " << cp.k_drafted << ",\n";
                ofs << "    \"t_draft_us\": " << cp.t_draft_us << ",\n";
                ofs << "    \"t_target_us\": " << cp.t_target_us << ",\n";
                ofs << "    \"t_process_us\": " << cp.t_process_us << ",\n";
                ofs << "    \"t_sample_us\": " << cp.t_sample_us << ",\n";
                ofs << "    \"t_rollback_us\": " << cp.t_rollback_us << ",\n";
                ofs << "    \"t_host_us\": " << cp.t_host_us << ",\n";
                ofs << "    \"t_cycle_us\": " << cp.t_cycle_us << ",\n";
                ofs << "    \"n_drafted\": " << cp.n_drafted << ",\n";
                ofs << "    \"n_accepted\": " << cp.n_accepted << ",\n";
                ofs << "    \"n_generated\": " << cp.n_generated << ",\n";
                ofs << "    \"accepted_positions\": [";
                for (size_t j = 0; j < cp.accepted_positions.size(); ++j) {
                    ofs << cp.accepted_positions[j] << (j + 1 < cp.accepted_positions.size() ? ", " : "");
                }
                ofs << "],\n";
                ofs << "    \"c\": [";
                for (size_t j = 0; j < cp.c.size(); ++j) {
                    ofs << cp.c[j] << (j + 1 < cp.c.size() ? ", " : "");
                }
                ofs << "],\n";
                ofs << "    \"a\": [";
                for (size_t j = 0; j < cp.a.size(); ++j) {
                    ofs << cp.a[j] << (j + 1 < cp.a.size() ? ", " : "");
                }
                ofs << "]\n";
                ofs << "  }" << (i + 1 < cycle_profiles.size() ? ",\n" : "\n");
            }
            ofs << "]\n";
            ofs.close();
            LOG_INF("[spec-cycle-log] Wrote %zu cycles to %s\n", cycle_profiles.size(), env_cycle_log);
        } else {
            LOG_ERR("[spec-cycle-log] Failed to open %s for writing\n", env_cycle_log);
        }
    }

    llama_batch_free(batch_tgt);

    common_speculative_free(spec);

    llama_backend_free();

    LOG("\n\n");

    return 0;
}
