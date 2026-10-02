#include "speculative.h"

#include "../src/llama-ext.h"
#include "common.h"
#include "ggml-cpp.h"
#include "ggml.h"
#include "llama.h"

#include <cmath>
#include <limits>
#ifdef LLAMA_DSPARK_MARKOV_CUDA
#    include "dspark-markov.h"
#endif
#ifdef LLAMA_DSPARK_MARKOV_METAL
#    include "dspark-markov-metal.h"
#endif
#ifdef LLAMA_DSPARK_MARKOV_BLAS
#    include <cblas.h>
#endif
#include "../src/llama-ext.h"  // staging API: llama_set_embeddings_nextn / llama_get_embeddings_nextn_ith (used by MTP)
#include "log.h"
#include "ngram-cache.h"
#include "ngram-map.h"
#include "ngram-mod.h"
#include "sampling.h"
#include "spec-arbiter.h"
#include "speculative-tap.h"

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <map>
#include <mutex>
#include <thread>
#include <utility>

#define SPC_DBG(fmt, ...) LOG_DBG("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_TRC(fmt, ...) LOG_TRC("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_INF(fmt, ...) LOG_INF("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_WRN(fmt, ...) LOG_WRN("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_ERR(fmt, ...) LOG_ERR("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_CNT(fmt, ...) LOG_CNT(""              fmt,               __VA_ARGS__)

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

const std::map<std::string, common_speculative_type> common_speculative_type_from_name_map = {
    {"none",          COMMON_SPECULATIVE_TYPE_NONE},
    {"draft-simple",  COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE},
    {"draft-eagle3",  COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3},
    {"draft-mtp",     COMMON_SPECULATIVE_TYPE_DRAFT_MTP},
    {"draft-dflash",  COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH},
    {"draft-dspark",  COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK},
    {"ngram-simple",  COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE},
    {"ngram-map-k",   COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K},
    {"ngram-map-k4v", COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V},
    {"ngram-mod",     COMMON_SPECULATIVE_TYPE_NGRAM_MOD},
    {"ngram-cache",   COMMON_SPECULATIVE_TYPE_NGRAM_CACHE}
};

static std::string common_speculative_get_devices_str(const std::vector<ggml_backend_dev_t> & devices) {
    std::string result;
    for (size_t i = 0; i < devices.size(); i++) {
        if (devices[i] == nullptr) {
            continue;
        }
        if (!result.empty()) result += ", ";
        result += ggml_backend_dev_name(devices[i]);
    }
    return result.empty() ? "default" : result;
}

struct common_speculative_config {
    common_speculative_type type;
    common_params_speculative params;

    common_speculative_config(common_speculative_type t,
            const common_params_speculative & p = common_params_speculative{}) : type(t), params(p) {}
};

static bool common_speculative_are_compatible(
    const llama_model * model_tgt,
    const llama_model * model_dft) {
    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    const auto vocab_type_tgt = llama_vocab_type(vocab_tgt);
    SPC_DBG("vocab_type tgt: %d\n", vocab_type_tgt);

    const auto vocab_type_dft = llama_vocab_type(vocab_dft);
    SPC_DBG("vocab_type dft: %d\n", vocab_type_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        SPC_WRN("draft model vocab type must match target model to use speculation but "
                "vocab_type_dft = %d while vocab_type_tgt = %d\n", vocab_type_dft, vocab_type_tgt);
        return false;
    }

    if (llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        (llama_vocab_get_add_bos(vocab_tgt) && llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft))) {
        SPC_WRN("draft model bos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                llama_vocab_get_add_bos(vocab_tgt), llama_vocab_get_add_bos(vocab_dft),
                llama_vocab_bos(vocab_tgt), llama_vocab_bos(vocab_dft));
        return false;
    }

    if (llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        (llama_vocab_get_add_eos(vocab_tgt) && llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft))) {
        SPC_WRN("draft model eos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                llama_vocab_get_add_eos(vocab_tgt), llama_vocab_get_add_eos(vocab_dft),
                llama_vocab_eos(vocab_tgt), llama_vocab_eos(vocab_dft));
        return false;
    }

    {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft
            ? n_vocab_tgt - n_vocab_dft
            : n_vocab_dft - n_vocab_tgt;

        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            SPC_DBG("draft model vocab must closely match target model to use speculation but "
                    "target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_vocab_n_tokens(vocab_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return false;
        }

        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
            const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);

            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                SPC_DBG("draft model vocab must match target model to use speculation but "
                        "token %d content differs - target '%s', draft '%s'\n", i,
                        common_token_to_piece(vocab_tgt, i).c_str(),
                        common_token_to_piece(vocab_dft, i).c_str());
                return false;
            }
        }
    }

    return true;
}

using common_speculative_draft_params_vec = std::vector<common_speculative_draft_params>;

// state of an implementation of speculative decoding
//
// each implementation has a unique type and a state that is implementation-specific
// in a subclass of common_speculative_impl
struct common_speculative_impl {
    const common_speculative_type type;

    uint32_t n_seq;

    size_t n_call_begin  = 0; // number of times this implementation was called for refresh.
    size_t n_call_draft  = 0; // number of times this implementation was called for generation.
    size_t n_call_accept = 0; // number of times this implementation was called for accumulation.

    size_t n_gen_drafts = 0; // number of times a draft or part was generated by this implementation.
    size_t n_acc_drafts = 0; // number of times a draft or part was accepted by the target model.
    size_t n_gen_tokens = 0; // number of tokens generated by this implementation.
    size_t n_acc_tokens = 0; // number of tokens accepted by the target model.

    std::vector<size_t> n_acc_tokens_per_pos; // number of tokens accepted per draft position.

    // TODO: track performance of most recent calls
    const bool gen_perf = true; // whether to generate performance stats.

    int64_t t_begin_us  = 0; // total time spent in refresh of this implementation in microseconds.
    int64_t t_draft_us  = 0; // total time spent in generating drafts in this implementation in microseconds.
    int64_t t_accept_us = 0; // total time spent in accumulation of this implementation in microseconds.

    common_speculative_impl(common_speculative_type type, uint32_t n_seq) : type(type), n_seq(n_seq) {}

    virtual ~common_speculative_impl() = default;

    virtual bool need_embd() const { return false; }

    virtual bool need_embd_nextn() const { return false; }

    virtual bool need_embd_capture() const { return false; }

    virtual bool stage_test_ctx_feat(llama_seq_id, const float *, int64_t, int64_t, const int32_t *) { return false; }

    virtual void begin(llama_seq_id seq_id, const llama_tokens & prompt) = 0;

    virtual bool process(const llama_batch & batch) = 0;

    virtual void draft(common_speculative_draft_params_vec & dparams) = 0;

    virtual void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) = 0;

    // (optional) serialize/restore per-seq internal state (e.g. eagle3's deferred boundary).
    virtual bool get_state(llama_seq_id /*seq_id*/, std::vector<uint8_t> & /*data*/) const { return false; }
    virtual void set_state(llama_seq_id /*seq_id*/, const std::vector<uint8_t> & /*data*/) {}
};

struct common_speculative_impl_draft_dspark : public common_speculative_impl {
    common_params_speculative_draft params;  // reuses the draft-model params slot (ctx_tgt/ctx_dft)

    int64_t n_embd        = 0;
    int64_t n_vocab       = 0;  // from token_embd's own shape; dspark has no tokenizer/vocab of its own
    int64_t n_capture     = 0;  // target_layer_ids count
    int64_t n_embd_cap    = 0;  // n_capture * n_embd (raw pre-fc tap width)
    int32_t block_size    = 0;
    int32_t mask_token_id = 0;
    int64_t draft_window  = 0;

    // vanilla Markov head weights, host-resident (loaded once at construction
    // via llama_model_dspark_get_markov): [n_vocab * n_rank] row-major, rank
    // fastest-varying. See the resample loop in draft() for how these are used.
    std::vector<float>                                                         markov_w1;
    std::vector<float>                                                         markov_w2;
    std::vector<float>                                                         markov_bias;
    int64_t                                                                    markov_rank = 0;
    bool                                                                       has_markov  = false;
    std::vector<std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>> graph_samplers;
    int64_t                                                                    markov_time_us    = 0;
    int64_t                                                                    markov_calls      = 0;
    bool                                                                       dcut_enabled      = false;
    int32_t                                                                    draft_rows        = 0;
    int32_t                                                                    correction_rows   = 0;
    bool                                                                       correction_prefix = false;
    float                                                                      dcut_costs[4]     = {};
#ifdef LLAMA_DSPARK_MARKOV_CUDA
    std::unique_ptr<dspark_markov_cuda, decltype(&dspark_markov_cuda_free)> markov_cuda{ nullptr,
                                                                                         dspark_markov_cuda_free };
#endif
#ifdef LLAMA_DSPARK_MARKOV_METAL
    std::unique_ptr<dspark_markov_metal, decltype(&dspark_markov_metal_free)> markov_metal{ nullptr,
                                                                                            dspark_markov_metal_free };
#endif

    llama_batch batch;  // ctx_dft batch; no embd channel -- context features are
                        // staged out-of-band via llama_set_dspark_ctx, not batch.embd

    // --- per-seq persistent state --------------------------------------
    // Absolute end of committed draft context, not the number of resident rows.
    std::vector<int64_t> n_cache;

    // growing buffer of not-yet-consumed target-tap context rows, accumulated
    // across process() calls since the last draft() call drained them. Rows
    // are contiguous and strictly increasing in position (asserted in draft()).
    std::vector<std::vector<float>>   ctx_feat;  // [n_seq][rows * n_embd_cap]
    std::vector<std::vector<int32_t>> ctx_pos;   // [n_seq][rows]

    // how many of the currently-buffered rows were appended since the last
    // accept() call. accept() trims exactly this many down to n_accepted+1,
    // discarding the rejected tail, leaving any earlier
    // (already-accepted-but-not-yet-drained) rows untouched. This is what
    // lets dspark's context stay correct even on rounds where a DIFFERENT
    // implementation's draft is the one that gets verified: process() runs
    // (and accumulates) unconditionally for every registered impl, and
    // accept() runs on every impl too (is_other=true for the ones that didn't
    // draft), so dspark's own bookkeeping tracks the real generation stream
    // regardless of who proposed a given round's tokens.
    std::vector<int64_t> rows_since_accept;

    // process()'s per-seq contiguous-range bookkeeping (mirrors draft-mtp).
    std::vector<int32_t> i_batch_beg;
    std::vector<int32_t> i_batch_end;

    common_speculative_impl_draft_dspark(const common_params_speculative & params, uint32_t n_seq) :
        common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK, n_seq),
        params(params.draft) {
        auto * ctx_dft = this->params.ctx_dft;
        auto * ctx_tgt = this->params.ctx_tgt;
        GGML_ASSERT(ctx_dft && ctx_tgt && "dspark requires ctx_tgt and ctx_dft to be set");

        const llama_model * model_dft = llama_get_model(ctx_dft);

        llama_dspark_meta meta;
        if (!llama_model_dspark_get_meta(model_dft, &meta)) {
            throw std::runtime_error(
                "dspark: ctx_dft's model does not look like a dspark drafter (missing dspark.*.block_size KV)");
        }

        n_embd        = meta.n_embd;
        n_vocab       = meta.n_vocab;
        n_capture     = meta.n_capture;
        n_embd_cap    = meta.n_embd_cap;
        block_size    = meta.block_size;
        mask_token_id = meta.mask_token_id;
        markov_rank   = meta.markov_rank;

        if (const char * value = std::getenv("DSPARK_DRAFT_WINDOW")) {
            char * end   = nullptr;
            draft_window = std::strtol(value, &end, 10);
            if (end == value || *end || draft_window < 0 || draft_window > std::numeric_limits<llama_pos>::max()) {
                throw std::runtime_error("DSPARK_DRAFT_WINDOW must be a nonnegative position count");
            }
        }
        LOG_INF("%s: draft_window=%lld (0=full prefix; target cache unchanged)\n", __func__, (long long) draft_window);

        has_markov =
            !meta.graph_corrected && markov_rank > 0 && llama_model_dspark_get_markov(model_dft, markov_w1, markov_w2);
        draft_rows      = block_size;
        correction_rows = block_size;
        if (const char * value = std::getenv("LLAMA_DSPARK_CORRECTION_PREFIX")) {
            char *     end  = nullptr;
            const long keep = std::strtol(value, &end, 10);
            if (end == value || *end || keep < 1 || keep > block_size || !meta.graph_corrected || n_seq != 1 ||
                params.draft.n_max != keep) {
                throw std::runtime_error(
                    "DSpark correction prefix requires graph correction, one sequence, and matching n_max");
            }
            correction_rows   = (int32_t) keep;
            correction_prefix = true;
        }
        const std::pair<const char *, int32_t *> row_options[] = {
            { "DSPARK_FORWARD_ROWS", &draft_rows },
            { "DSPARK_CORRECTION_ROWS", &correction_rows },
        };
        for (const auto & item : row_options) {
            if (const char * value = std::getenv(item.first)) {
                char *     end  = nullptr;
                const long rows = std::strtol(value, &end, 10);
                if (end == value || *end || rows < 1 || rows > block_size || !has_markov) {
                    throw std::runtime_error("DSpark experimental row count requires legacy Markov and 1..block_size");
                }
                *item.second = (int32_t) rows;
            }
        }
        if (correction_rows > draft_rows) {
            throw std::runtime_error("DSpark correction rows exceed forward rows");
        }
        LOG_INF("dspark: full_block=%d forward_rows=%d correction_rows=%d\n", block_size, draft_rows,
                correction_rows);
        if (n_vocab > std::numeric_limits<int>::max()) {
            throw std::runtime_error("dspark: vocab size exceeds cblas integer range");
        }
        markov_bias.resize((size_t) n_vocab);

        const char * cuda_mode = std::getenv("LLAMA_DSPARK_MARKOV_CUDA");
        if (const char * value = std::getenv("DSPARK_DCUT_COSTS")) {
            const char * next = value;
            for (int i = 0; i < 4; ++i) {
                char * end    = nullptr;
                dcut_costs[i] = std::strtof(next, &end);
                if (end == next || !std::isfinite(dcut_costs[i]) || dcut_costs[i] <= 0 ||
                    (i < 3 ? *end != ',' : *end != '\0')) {
                    throw std::runtime_error(
                        "DSPARK_DCUT_COSTS requires four positive finite comma-separated round costs");
                }
                next = end + (i < 3);
            }
            if (n_seq != 1 || block_size != 4 || !has_markov || !cuda_mode || std::strcmp(cuda_mode, "1")) {
                throw std::runtime_error("Experimental D-cut requires c1 block-4 legacy CUDA Markov");
            }
            dcut_enabled = true;
            LOG_INF(
                "dspark: dcut_policy=device_probability_cost_v1 costs=%.9g,%.9g,%.9g,%.9g "
                "score=uncalibrated_draft_probability\n",
                dcut_costs[0], dcut_costs[1], dcut_costs[2], dcut_costs[3]);
        }
        const char * metal_mode = std::getenv("LLAMA_DSPARK_MARKOV_METAL");
        if (metal_mode && std::strcmp(metal_mode, "0") && std::strcmp(metal_mode, "1")) {
            throw std::runtime_error("LLAMA_DSPARK_MARKOV_METAL must be 0 or 1");
        }
        const bool want_metal = metal_mode && std::strcmp(metal_mode, "1") == 0;
        if (want_metal && cuda_mode && std::strcmp(cuda_mode, "1") == 0) {
            throw std::runtime_error("Choose either Metal or CUDA Markov, not both");
        }
        if (cuda_mode && std::strcmp(cuda_mode, "0") && std::strcmp(cuda_mode, "1")) {
            throw std::runtime_error("LLAMA_DSPARK_MARKOV_CUDA must be 0 or 1");
        }
        if (cuda_mode && std::strcmp(cuda_mode, "1") == 0) {
            if (!has_markov) {
                throw std::runtime_error(
                    "CUDA Markov requested without a legacy Markov head; graph-corrected drafters use their own path");
            }
#ifdef LLAMA_DSPARK_MARKOV_CUDA
            markov_cuda.reset(
                dspark_markov_cuda_init(markov_w1.data(), markov_w2.data(), n_vocab, markov_rank, mask_token_id));
            if (!markov_cuda) {
                throw std::runtime_error("CUDA Markov initialization failed; refusing CPU fallback");
            }
            LOG_INF("dspark: correction_backend=CUDA_MARKOV\n");
#else
            throw std::runtime_error("CUDA Markov requested but not compiled");
#endif
        } else if (!want_metal) {
            LOG_INF("dspark: correction_backend=%s\n", meta.graph_corrected ? "DRAFT_GRAPH" : "HOST");
        }
        if (want_metal) {
            if (!has_markov) {
                throw std::runtime_error(
                    "Metal Markov requires a legacy Markov head; graph-corrected drafters use their own path");
            }
#ifdef LLAMA_DSPARK_MARKOV_METAL
            markov_metal.reset(
                dspark_markov_metal_init(markov_w1.data(), markov_w2.data(), n_vocab, markov_rank, mask_token_id));
            if (!markov_metal) {
                throw std::runtime_error("Metal Markov initialization failed; refusing CPU fallback");
            }
            LOG_INF("dspark: correction_backend=METAL_MARKOV\n");
#else
            throw std::runtime_error("Metal Markov requested but not compiled");
#endif
        }

        LOG_INF("%s: adding speculative implementation 'draft-dspark'\n", __func__);
        LOG_INF(
            "%s: - block_size=%d, mask_token_id=%d, n_capture=%lld, n_embd=%lld, n_vocab=%lld, markov_rank=%lld, "
            "has_markov=%d\n",
            __func__, block_size, mask_token_id, (long long) n_capture, (long long) n_embd, (long long) n_vocab,
            (long long) markov_rank, (int) has_markov);
        if (markov_rank > 0 && !has_markov && !meta.graph_corrected) {
            LOG_WRN(
                "%s: dspark model reports markov_rank=%lld but its markov head weights could not be read "
                "(gated/rnn markov head type? only 'vanilla' is supported) -- "
                "block logits will NOT be markov-corrected\n",
                __func__, (long long) markov_rank);
        }

        // dspark attention is fully non-causal within a call: the draft block
        // attends over the WHOLE persistent cache plus itself, with no
        // position-based masking (attention_mask=None, is_causal=False in the
        // reference) -- see src/models/dspark.cpp's header comment.
        llama_set_causal_attn(ctx_dft, false);

        const char * greedy_env = std::getenv("LLAMA_DSPARK_GREEDY_IDS");
        if (greedy_env && std::strcmp(greedy_env, "0") && std::strcmp(greedy_env, "1")) {
            throw std::runtime_error("LLAMA_DSPARK_GREEDY_IDS must be 0 or 1");
        }
        if (greedy_env && std::strcmp(greedy_env, "1") == 0) {
            if (!meta.graph_corrected) {
                throw std::runtime_error("DSpark greedy IDs require graph correction");
            }
            graph_samplers.reserve(n_seq);
            for (llama_seq_id seq = 0; seq < (llama_seq_id) n_seq; ++seq) {
                std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> chain(
                    llama_sampler_chain_init(llama_sampler_chain_default_params()), llama_sampler_free);
                llama_sampler_chain_add(chain.get(), llama_sampler_init_greedy());
                graph_samplers.push_back(std::move(chain));
            }
        }

        n_cache.assign(n_seq, 0);
        ctx_feat.assign(n_seq, {});
        ctx_pos.assign(n_seq, {});
        rows_since_accept.assign(n_seq, 0);
        i_batch_beg.assign(n_seq, -1);
        i_batch_end.assign(n_seq, -1);

        const int32_t n_b      = (int32_t) llama_n_batch(ctx_dft);
        batch                  = llama_batch_init(/* n_tokens = */ n_b, /* embd = */ 0, /* n_seq_max = */ 1);
        llama_seq_id attaching = 0;
        try {
            for (; attaching < (llama_seq_id) graph_samplers.size(); ++attaching) {
                if (!llama_set_sampler(ctx_dft, attaching, graph_samplers[attaching].get())) {
                    throw std::runtime_error("DSpark greedy ID offload failed; refusing host fallback");
                }
            }
        } catch (...) {
            // Detach before member destruction frees the samplers, including the failed attachment.
            for (llama_seq_id seq = 0; seq <= attaching && seq < (llama_seq_id) graph_samplers.size(); ++seq) {
                llama_set_sampler(ctx_dft, seq, nullptr);
            }
            llama_batch_free(batch);
            throw;
        }
        if (!graph_samplers.empty()) {
            LOG_INF("dspark: output_backend=GRAPH_GREEDY_IDS\n");
        }
    }

    ~common_speculative_impl_draft_dspark() override {
        for (llama_seq_id seq = 0; seq < (llama_seq_id) graph_samplers.size(); ++seq) {
            llama_set_sampler(params.ctx_dft, seq, nullptr);
        }
        LOG_INF("dspark: correction_calls=%lld correction_wall_us=%lld\n", (long long) markov_calls,
                (long long) markov_time_us);
        llama_batch_free(batch);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & /*prompt*/) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        // fresh generation: drop any leftover state from a prior generation
        // that reused this seq slot, and make sure ctx_dft's own cache for
        // this seq starts empty.
        n_cache[seq_id] = 0;
        ctx_feat[seq_id].clear();
        ctx_pos[seq_id].clear();
        rows_since_accept[seq_id] = 0;

        llama_memory_seq_rm(llama_get_memory(params.ctx_dft), seq_id, 0, -1);
    }

    bool process(const llama_batch & batch_in) override {
        if (batch_in.n_tokens <= 0) {
            return true;
        }

        // TODO: how to make it work with vision tokens? (mirrors draft-mtp)
        if (batch_in.token == nullptr || batch_in.embd != nullptr) {
            return true;
        }

        const int32_t n_tokens = batch_in.n_tokens;

        std::fill(i_batch_beg.begin(), i_batch_beg.end(), -1);
        std::fill(i_batch_end.begin(), i_batch_end.end(), -1);

        for (int k = 0; k < n_tokens; ++k) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                GGML_ASSERT(batch_in.n_seq_id[k] == 1);

                if (batch_in.seq_id[k][0] == seq_id) {
                    i_batch_end[seq_id] = k;
                    if (i_batch_beg[seq_id] < 0) {
                        i_batch_beg[seq_id] = k;
                    }
                }
            }
        }

        auto * ctx_tgt = params.ctx_tgt;

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_batch_beg[seq_id] < 0) {
                continue;
            }

            const int32_t n_rows = i_batch_end[seq_id] - i_batch_beg[seq_id] + 1;

            auto & feat = ctx_feat[seq_id];
            auto & pos  = ctx_pos[seq_id];

            const size_t row0 = pos.size();
            feat.resize((row0 + (size_t) n_rows) * (size_t) n_embd_cap);
            pos.resize(row0 + (size_t) n_rows);

            for (int32_t i = 0; i < n_rows; ++i) {
                const int32_t k = i_batch_beg[seq_id] + i;

                // NOTE: capture rows always use the masked (output-row) layout
                // (see src/llama-context.cpp's get_embeddings_capture_ith) --
                // this requires the caller to have requested logits/output on
                // EVERY row it wants a capture row for (unlike the pre-norm
                // path MTP uses, which can force unmasked extraction). For a
                // long prompt this means every prefill row, not just the
                // last -- a caller-side (main/server driver loop) requirement
                // when a registered impl reports need_embd_capture(), exactly
                // analogous to draft-mtp's own begin()-time warning about
                // need_embd_nextn.
                const float * cap = llama_get_embeddings_capture_ith(ctx_tgt, k);
                if (cap == nullptr) {
                    LOG_ERR(
                        "%s: llama_get_embeddings_capture_ith(%d) returned null -- was "
                        "llama_set_capture_layers() engaged and logits requested for every "
                        "row this impl needs?\n",
                        __func__, k);
                    return false;
                }

                std::memcpy(feat.data() + (row0 + (size_t) i) * (size_t) n_embd_cap, cap,
                            (size_t) n_embd_cap * sizeof(float));
                pos[row0 + i] = batch_in.pos[k];
            }

            rows_since_accept[seq_id] += n_rows;
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        const char * prefix_value = std::getenv("LLAMA_DSPARK_CORRECTION_PREFIX");
        if (correction_prefix) {
            char *     end  = nullptr;
            const long keep = prefix_value ? std::strtol(prefix_value, &end, 10) : 0;
            if (!prefix_value || end == prefix_value || *end || keep != correction_rows) {
                throw std::runtime_error("DSpark correction prefix changed after initialization");
            }
        } else if (prefix_value) {
            throw std::runtime_error("DSpark correction prefix enabled after initialization");
        }
        auto *        ctx_dft     = params.ctx_dft;
        const int64_t n_batch_max = (int64_t) llama_n_batch(ctx_dft);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            auto & feat = ctx_feat[seq_id];
            auto & pos  = ctx_pos[seq_id];

            int64_t       L       = n_cache[seq_id];
            const int64_t start   = dp.n_past;
            int64_t       ctx_len = start - L;

            if (ctx_len <= 0) {
                LOG_WRN(
                    "%s: seq %d has no new context rows staged (n_past=%lld, cache=%lld) -- "
                    "skipping this round\n",
                    __func__, (int) seq_id, (long long) start, (long long) L);
                continue;
            }
            if ((int64_t) pos.size() != ctx_len) {
                LOG_ERR(
                    "%s: seq %d staged context rows (%zu) != expected ctx_len (%lld) -- "
                    "n_past bookkeeping is out of sync with process()/accept(); "
                    "aborting draft for this seq this round\n",
                    __func__, (int) seq_id, pos.size(), (long long) ctx_len);
                continue;
            }
            GGML_ASSERT(pos.front() == (int32_t) L &&
                        "dspark: staged rows do not start at the drafter's cache position");
            GGML_ASSERT(pos.back() == (int32_t) start - 1 &&
                        "dspark: staged rows do not end just before the anchor position");

            // Evict only draft context. Preserve absolute RoPE positions and the commit cursor.
            const int64_t window_begin = draft_window > 0 ? std::max(int64_t(0), start - draft_window) : 0;
            const int64_t skip         = std::max(int64_t(0), window_begin - L);
            L += skip;
            ctx_len -= skip;

            const int64_t chunk_capacity = std::min(n_batch_max, (int64_t) llama_n_ubatch(ctx_dft)) - draft_rows;
            if (chunk_capacity < 1) {
                throw std::runtime_error("dspark: batch cannot hold context and draft block");
            }

            if (draft_window > 0 &&
                !llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, 0, (llama_pos) window_begin)) {
                throw std::runtime_error("dspark: draft window eviction failed");
            }
            for (int64_t offset = 0; offset < ctx_len;) {
                const int64_t count     = std::min(chunk_capacity, ctx_len - offset);
                const int64_t chunk_end = L + offset + count;
                llama_set_dspark_ctx(ctx_dft, feat.data() + (skip + offset) * n_embd_cap, count, n_embd_cap,
                                     pos.data() + skip + offset);
                common_batch_clear(batch);
                for (int64_t i = 0; i < count; ++i) {
                    common_batch_add(batch, 0, (llama_pos) (L + offset + i), { seq_id }, false);
                }
                // Context K/V depend only on target features. Intermediate block outputs are discarded.
                common_batch_add(batch, dp.id_last, (llama_pos) chunk_end, { seq_id }, true);
                for (int32_t k = 1; k < draft_rows; ++k) {
                    common_batch_add(batch, mask_token_id, (llama_pos) (chunk_end + k), { seq_id },
                                     !correction_prefix || k < correction_rows);
                }
                const int32_t rc = llama_decode(ctx_dft, batch);
                llama_set_dspark_ctx(ctx_dft, nullptr, 0, 0, nullptr);
                if (rc != 0) {
                    throw std::runtime_error("dspark: chunk decode failed");
                }
                if (!llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, (llama_pos) chunk_end, -1)) {
                    throw std::runtime_error("dspark: draft tail removal failed");
                }
                offset += count;
            }
            if (draft_window > 0) {
                const auto mem = llama_get_memory(ctx_dft);
                if (llama_memory_seq_pos_min(mem, seq_id) != window_begin ||
                    llama_memory_seq_pos_max(mem, seq_id) != start - 1) {
                    throw std::runtime_error("dspark: draft window position invariant failed");
                }
            }
            n_cache[seq_id] = start;

            feat.clear();
            pos.clear();
            rows_since_accept[seq_id] = 0;  // this round's rows were just consumed

            // --- sequential Markov resample -------------------------------
            // step_logits[k] = base_logits[k] + markov_w2(markov_w1(prev_token)),
            // where prev_token is the block's own anchor token for k==0 and the
            // ACTUALLY SAMPLED token from step k-1 for k>0. This must never be
            // batched over mask_token_id for all block positions at once --
            // that exact bug class already hit the on-device (MLX/Swift) port.
            // The assert below makes the sequential dependency structural
            // rather than just a comment: it is unsatisfiable if this loop is
            // ever refactored to precompute prev_token_ids up front from
            // draft_input_ids instead of chaining the sampled result forward.
            llama_tokens result;
            result.reserve(block_size);
            if (!graph_samplers.empty()) {
                for (int32_t k = 0; k < correction_rows; ++k) {
                    const int32_t     output_rows = correction_prefix ? correction_rows : draft_rows;
                    const llama_token token       = llama_get_sampled_token_ith(ctx_dft, -output_rows + k);
                    if (token < 0 || token >= n_vocab || token == mask_token_id) {
                        throw std::runtime_error("DSpark graph returned an invalid draft token");
                    }
                    result.push_back(token);
                }
                if (result.size() >= (size_t) params.n_min) {
                    *dp.result = std::move(result);
                }
                continue;
            }

            // dense output buffer: only the block_size draft rows requested
            // logits this call, so llama_get_logits() is already exactly
            // block_size*n_vocab floats in row order -- no per-row index
            // resolution needed (mirrors tests/test-dspark-forward.cpp's
            // llama_get_logits(ctx) usage). llama_get_logits_ith(ctx, i)
            // would need i to be the RAW ubatch row (ctx_len + k here), since
            // it resolves through output_resolve_row() same as the
            // pre-norm/capture accessors -- the bulk buffer sidesteps that.
            const float * logits_base = llama_get_logits(ctx_dft);
            if (logits_base == nullptr) {
                LOG_ERR("%s: llama_get_logits(ctx_dft) returned null for seq %d\n", __func__, (int) seq_id);
                continue;
            }

            llama_token   prev_token          = dp.id_last;
            const int64_t correction_start_us = ggml_time_us();

#ifdef LLAMA_DSPARK_MARKOV_METAL
            if (markov_metal) {
                result.resize(correction_rows);
                if (!dspark_markov_metal_resample(markov_metal.get(), logits_base, dp.id_last, correction_rows,
                                                  result.data())) {
                    throw std::runtime_error("Metal Markov resample failed; refusing CPU fallback");
                }
                for (llama_token token : result) {
                    if (token < 0 || token >= n_vocab || token == mask_token_id) {
                        throw std::runtime_error("Metal Markov returned an invalid draft token");
                    }
                }
                markov_time_us += ggml_time_us() - correction_start_us;
                ++markov_calls;
                if (result.size() >= (size_t) params.n_min) {
                    *dp.result = std::move(result);
                }
                continue;
            }
#endif

#ifdef LLAMA_DSPARK_MARKOV_CUDA
            if (markov_cuda) {
                if (dcut_enabled && dp.n_max > 0 && dp.n_max < correction_rows) {
                    throw std::runtime_error("D-cut depth must not be overridden by a fixed verification cap");
                }
                result.resize(correction_rows);
                int32_t    retained = correction_rows;
                const bool ok = dcut_enabled ?
                                    dspark_markov_cuda_dcut(markov_cuda.get(), logits_base, dp.id_last, correction_rows,
                                                            result.data(), dcut_costs, &retained) :
                                    dspark_markov_cuda_resample(markov_cuda.get(), logits_base, dp.id_last,
                                                                correction_rows, result.data());
                if (!ok) {
                    throw std::runtime_error("CUDA Markov resample failed; refusing CPU fallback");
                }
                for (llama_token token : result) {
                    if (token < 0 || token >= n_vocab || token == mask_token_id) {
                        throw std::runtime_error("CUDA Markov returned an invalid draft token");
                    }
                }
                markov_time_us += ggml_time_us() - correction_start_us;
                ++markov_calls;
                result.resize(retained);
                if (result.size() >= (size_t) params.n_min) {
                    *dp.result = std::move(result);
                }
                continue;
            }
#endif

            for (int32_t k = 0; k < correction_rows; ++k) {
                if (k > 0) {
                    GGML_ASSERT(prev_token != mask_token_id &&
                                "dspark: markov resample must chain the previous step's SAMPLED "
                                "token, never mask_token_id -- do not batch this over the block");
                }

                const float * base_logits = logits_base + (size_t) k * n_vocab;

                llama_token best_id = mask_token_id == 0 ? 1 : 0;
                float       best_v  = -std::numeric_limits<float>::infinity();

                if (has_markov) {
                    const float * emb = markov_w1.data() + (size_t) prev_token * (size_t) markov_rank;
#ifdef LLAMA_DSPARK_MARKOV_BLAS
                    cblas_sgemv(CblasRowMajor, CblasNoTrans, (int) n_vocab, (int) markov_rank, 1.0f, markov_w2.data(),
                                (int) markov_rank, emb, 1, 0.0f, markov_bias.data(), 1);

                    for (int64_t v = 0; v < n_vocab; ++v) {
                        if (v == mask_token_id) {
                            continue;
                        }
                        const float logit = base_logits[v] + markov_bias[(size_t) v];
                        if (logit > best_v) {
                            best_v  = logit;
                            best_id = (llama_token) v;
                        }
                    }
#else
                    for (int64_t v = 0; v < n_vocab; ++v) {
                        if (v == mask_token_id) {
                            continue;
                        }
                        const float * w2row = markov_w2.data() + (size_t) v * (size_t) markov_rank;
                        float         bias  = 0.0f;
                        for (int64_t r = 0; r < markov_rank; ++r) {
                            bias += emb[r] * w2row[r];
                        }
                        const float logit = base_logits[v] + bias;
                        if (logit > best_v) {
                            best_v  = logit;
                            best_id = (llama_token) v;
                        }
                    }
#endif
                } else {
                    for (int64_t v = 0; v < n_vocab; ++v) {
                        if (v == mask_token_id) {
                            continue;
                        }
                        if (base_logits[v] > best_v) {
                            best_v  = base_logits[v];
                            best_id = (llama_token) v;
                        }
                    }
                }

                result.push_back(best_id);
                prev_token = best_id;  // chain the SAMPLED token, never mask_token_id
            }

            markov_time_us += ggml_time_us() - correction_start_us;
            ++markov_calls;
            if (result.size() < (size_t) params.n_min) {
                continue;  // dp.result stays empty: treated as a failed draft this round
            }

            *dp.result = std::move(result);
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool /*is_other*/) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        const int64_t n_round_rows = rows_since_accept[seq_id];
        rows_since_accept[seq_id]  = 0;
        if (n_round_rows <= 0) {
            return;
        }

        // process() (or the test-only injection hook) unconditionally
        // captured tap features for the WHOLE verify batch, including any
        // positions past the accepted prefix; trim this round's
        // freshly-appended tail down to n_accepted+1 rows (the actually
        // committed context), discarding the rejected continuation. This
        // mirrors the DeepSpec reference's evaluator._update():
        //   context.target_hidden_states = verified_target_hidden[:, :accepted_draft_tokens+1, :]
        // Runs the same way regardless of is_other: dspark's own context must
        // stay correct even on rounds where a different implementation's
        // draft is the one that gets verified.
        const int64_t keep = std::min<int64_t>(n_round_rows, (int64_t) n_accepted + 1);
        const int64_t drop = n_round_rows - keep;

        if (drop > 0) {
            auto & feat = ctx_feat[seq_id];
            auto & pos  = ctx_pos[seq_id];

            const size_t total_rows = pos.size();
            GGML_ASSERT((int64_t) total_rows >= drop);

            feat.resize((total_rows - (size_t) drop) * (size_t) n_embd_cap);
            pos.resize(total_rows - (size_t) drop);
        }
    }

    bool need_embd() const override { return false; }

    bool need_embd_nextn() const override { return false; }

    bool need_embd_capture() const override { return true; }

    bool stage_test_ctx_feat(llama_seq_id    seq_id,
                             const float *   feat_in,
                             int64_t         n_rows,
                             int64_t         n_embd_cap_in,
                             const int32_t * pos_in) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return false;
        }
        if (n_embd_cap_in != n_embd_cap) {
            LOG_ERR("%s: n_embd_cap mismatch: got %lld, expected %lld\n", __func__, (long long) n_embd_cap_in,
                    (long long) n_embd_cap);
            return false;
        }
        if (n_rows <= 0) {
            return true;
        }

        auto & feat = ctx_feat[seq_id];
        auto & pos  = ctx_pos[seq_id];

        const size_t row0 = pos.size();
        feat.resize((row0 + (size_t) n_rows) * (size_t) n_embd_cap);
        pos.resize(row0 + (size_t) n_rows);

        std::memcpy(feat.data() + row0 * (size_t) n_embd_cap, feat_in,
                    (size_t) n_rows * (size_t) n_embd_cap * sizeof(float));
        std::memcpy(pos.data() + row0, pos_in, (size_t) n_rows * sizeof(int32_t));

        rows_since_accept[seq_id] += n_rows;
        return true;
    }
};

struct common_speculative_impl_draft_simple : public common_speculative_impl {
    common_params_speculative_draft params;

    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;

    common_speculative_impl_draft_simple(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE, n_seq)
        , params(params.draft)
    {
        auto * ctx_dft = this->params.ctx_dft;
        auto * ctx_tgt = this->params.ctx_tgt;

        if (!ctx_dft) {
            throw std::runtime_error("draft-simple requires a draft context");
        }

        SPC_TRC("%s", "adding speculative implementation 'draft-simple'\n");
        SPC_TRC("- n_max=%d, n_min=%d, p_min=%f\n", this->params.n_max, this->params.n_min, this->params.p_min);
        SPC_TRC("- gpu_layers=%d, cache_k=%s, cache_v=%s, ctx_tgt=%s, ctx_dft=%s, devices=[%s]\n",
                this->params.n_gpu_layers,
                ggml_type_name(this->params.cache_type_k),
                ggml_type_name(this->params.cache_type_v),
                ctx_tgt ? "yes" : "no",
                ctx_dft ? "yes" : "no",
                common_speculative_get_devices_str(this->params.devices).c_str());

        batch = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);

        // TODO: optimize or pass from outside?
        // {
        //     common_params_sampling params;
        //     params.no_perf = false;
        //
        //     params.top_k = 40;
        //     params.top_p = 0.9;
        //
        //     params.samplers = {
        //         COMMON_SAMPLER_TYPE_TOP_K,
        //         COMMON_SAMPLER_TYPE_TOP_P,
        //         COMMON_SAMPLER_TYPE_INFILL,
        //     };
        //
        //     result->smpl = common_sampler_init(llama_get_model(ctx_dft), params);
        // }

        smpls.resize(n_seq);
        for (auto & smpl : smpls) {
            common_params_sampling params;
            params.no_perf = false;
            params.top_k = 10;
            params.samplers = {
                COMMON_SAMPLER_TYPE_TOP_K,
            };

            smpl.reset(common_sampler_init(llama_get_model(ctx_dft), params));
        }

        const bool vocab_cmpt = common_speculative_are_compatible(llama_get_model(ctx_tgt), llama_get_model(ctx_dft));
        SPC_DBG("vocab_cmpt = %d\n", vocab_cmpt);

        if (!vocab_cmpt) {
            SPC_ERR("%s", "the target and draft vocabs are not compatible\n");

            throw std::runtime_error("draft model vocab type must match target model to use speculation");
        }

        if (n_seq != llama_n_seq_max(ctx_dft)) {
            SPC_ERR("n_seq mismatch: %d != %d\n", n_seq, llama_n_seq_max(ctx_dft));

            throw std::runtime_error("the draft model number of sequences is incompatible with the speculative n_seq");
        }
    }

    ~common_speculative_impl_draft_simple() override {
        llama_batch_free(batch);
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    bool process(const llama_batch & batch) override {
        auto * ctx_dft = params.ctx_dft;

        llama_batch batch_dft = batch;
        batch_dft.logits = nullptr;

        const int ret = llama_decode(ctx_dft, batch_dft);

        if (ret != 0) {
            SPC_ERR("failed to decode draft batch, ret = %d\n", ret);

            return false;
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            common_batch_add(batch, dp.id_last, dp.n_past, { seq_id }, true);
        }

        int ret = llama_decode(ctx_dft, batch);
        if (ret != 0) {
            SPC_ERR("llama_decode returned %d\n", ret);
            return;
        }

        int i = 0;

        while (n_drafting > 0) {
            int i_batch = 0;

            common_batch_clear(batch);

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                common_sampler_sample(smpl, ctx_dft, i_batch, true);
                ++i_batch;

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    SPC_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                // add drafted token for each sequence
                const llama_token id = cur_p->data[0].id;

                // only collect very high-confidence draft tokens
                if (cur_p->data[0].p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);

                if ((params.n_max <= (int) result.size()) ||
                    (dp.n_max > 0 && dp.n_max <= (int) result.size())) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                common_batch_add(batch, id, dp.n_past + i + 1, { seq_id }, true);
            }

            if (batch.n_tokens == 0) {
                break;
            }

            // evaluate the drafted tokens on the draft model
            ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                SPC_ERR("llama_decode[%d] returned %d\n", i, ret);
                break;
            }

            ++i;
        }

        for (auto & dp : dparams) {
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }
};


// EAGLE3 speculative decoding state
//
// Input of draft decoder: (This is different compared to MTP)
//   At "pos P", the decoder takes input pair (t_{P+1}, g_P), with RoPE at P.
//     - t_{P+1} = token at sequence pos P+1 (the *next* token after P)
//     - g_P     = encoder output = projection of target's extracted hidden states at P
//
// Deferred boundary (MTP doesn't have this issue):
//   Within a single process() call with n_tokens, we can only write decoder KV for
//   training pos 0..n_tokens-2. The last training pos (n_tokens-1) needs t_{n_tokens}
//   which lies *outside* this batch — it is the token target will sample next or the first token from next ubatch.
//   So the last training pos of each process() call is *deferred* to whichever next call has
//   the missing token in hand:
//     - multi-ubatch prefill: the next process()'s first token completes the pair
//                              (handled by the per-seq "cross-ubatch bridge")
//     - single-ubatch prefill / after verify: draft()'s seed step uses "dp.id_last"
//                              (target's freshest sample) to complete the pair
//
// Per-seq carry-over state:
//   pending_g_last    [n_embd_dec]  ┐  the deferred boundary's (g, pos). Set by
//   pending_pos_last  llama_pos     ┘  process() at end of ubatch (= last row);
//                                       rebased by accept() to first-non-accepted pos.
//   verify_g          [N × n_embd_dec] snapshot of process()'s encoder output;
//   verify_pos_first  llama_pos         consumed by accept() to recover the right
//   verify_g_rows     int32_t           pending_g_last row for any n_accepted value.
//
// Performance is overall good but there is waste in verify cycle:
//   process() runs encoder + decoder on the *full* verify batch including rows for
//   rejected drafts. The KV at those positions is then dropped.
//
// TODO: Not sure if we need optimization for this waste?
// If so we may need hybrid stash:
//      in verify mode, have process() only stash features and let draft() seed run
//      encoder+decoder on n_accepted+1 rows).
struct common_speculative_impl_draft_eagle3 : public common_speculative_impl {
    common_params_speculative_draft params;
    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;

    // backend sampler chain per seq, attached to ctx_dft
    std::vector<llama_sampler *> backend_chains;

    int32_t n_embd_dec = 0;       // draft context row width (n_embd_out: per-layer for DFly)
    int32_t n_embd_enc = 0;       // target_layer_ids_n * target_hidden_size
    int32_t n_embd_tgt = 0;       // target model hidden size
    int32_t n_layer_tgt = 0;      // target model layer count

    const int32_t * target_layer_ids   = nullptr; // model_dft's extract layer indices
    uint32_t        target_layer_ids_n = 0;

    // [per-seq] deferred boundary state
    std::vector<std::vector<float>> pending_g_last;
    std::vector<llama_pos>          pending_pos_last;

    // [per-seq] snapshot of the most recent process()'s encoder output
    std::vector<std::vector<float>> verify_g;         // [n_seq][n_rows * n_embd_dec]
    std::vector<llama_pos>          verify_pos_first; // [n_seq] — pos of verify_g[seq][0]
    std::vector<int32_t>            verify_g_rows;    // [n_seq] — number of rows

    // scratch buffer for concatenated target features [n_tokens, n_embd_enc]
    std::vector<float> features_buf;
    std::vector<float> g_embd_buf;

    common_speculative_impl_draft_eagle3(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3, n_seq)
        , params(params.draft)
    {
        SPC_TRC("%s", "adding speculative implementation 'draft-eagle3'\n");
        SPC_TRC("- n_max=%d, n_min=%d, p_min=%f, backend_sampling=%d\n", params.draft.n_max, params.draft.n_min, params.draft.p_min, (int) params.draft.backend_sampling);

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        GGML_ASSERT(ctx_tgt && ctx_dft && "EAGLE3 requires ctx_tgt and ctx_dft to be set");

        const llama_model * model_dft = llama_get_model(ctx_dft);
        const llama_model * model_tgt = llama_get_model(ctx_tgt);

        target_layer_ids   = llama_model_target_layer_ids  (model_dft);
        target_layer_ids_n = llama_model_target_layer_ids_n(model_dft);
        if (target_layer_ids_n != 3) {
            throw std::runtime_error("draft model is not eagle3 (expected 3 extract layers, got " +
                                     std::to_string(target_layer_ids_n) + ")");
        }

        n_embd_tgt = llama_model_n_embd(model_tgt);
        n_embd_dec = llama_model_n_embd_out(model_dft);
        n_embd_enc = (int32_t) target_layer_ids_n * n_embd_tgt;
        n_layer_tgt = llama_model_n_layer(model_tgt);

        const int32_t n_b = (int32_t) llama_n_batch(ctx_dft);
        batch = llama_batch_init(/*n_tokens=*/ n_b, /*embd=*/ n_embd_dec, /*n_seq_max=*/ 1);
        // llama_batch_init allocates only one of token/embd; eagle3 decoder needs both.
        // TODO: fix, how to call without malloc
        batch.token = (llama_token *) malloc(sizeof(llama_token) * n_b);

        smpls.resize(n_seq);
        for (auto & s : smpls) {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            sparams.top_k    = 10;
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            s.reset(common_sampler_init(llama_get_model(ctx_dft), sparams));
        }

        // offload draft sampling to the backend
        backend_chains.assign(n_seq, nullptr);
        if (this->params.backend_sampling) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(10));

                if (!llama_set_sampler(ctx_dft, seq_id, chain)) {
                    SPC_WRN("backend offload failed for seq_id=%d; using CPU sampler\n", (int) seq_id);
                    llama_sampler_free(chain);
                    chain = nullptr;
                }
                backend_chains[seq_id] = chain;
            }
        }

        // turn on extraction of the target layers' hidden states
        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            if (target_layer_ids[k] < n_layer_tgt) {
                llama_set_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k], true);
            } else if (target_layer_ids[k] == n_layer_tgt) {
                llama_set_embeddings_nextn(ctx_tgt, true, /*masked*/ false);
            } else {
                GGML_ABORT("EAGLE3: target layer id %d exceeds target n_layer %d", target_layer_ids[k], n_layer_tgt);
            }
        }

        // turn on extraction of the draft model's pre-norm hidden state
        // (used both for the encoder output g_embd and the decoder pre-norm output).
        llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ true);

        pending_g_last.assign(n_seq, std::vector<float>(n_embd_dec, 0.0f));
        pending_pos_last.assign(n_seq, -1);

        verify_g.assign(n_seq, std::vector<float>());
        verify_pos_first.assign(n_seq, -1);
        verify_g_rows.assign(n_seq, 0);
    }

    ~common_speculative_impl_draft_eagle3() override {
        auto * ctx_dft = this->params.ctx_dft;
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) backend_chains.size(); ++seq_id) {
            if (backend_chains[seq_id] == nullptr) {
                continue;
            }
            if (ctx_dft) {
                llama_set_sampler(ctx_dft, seq_id, nullptr);
            }
            llama_sampler_free(backend_chains[seq_id]);
        }
        backend_chains.clear();

        if (batch.token != nullptr) {
            free(batch.token);
            batch.token = nullptr;
        }
        llama_batch_free(batch);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }
        // expected state after prefill: ctx_dft has pos 0..N-2 (last position is deferred to
        // draft()'s seed step). Warn only if more than one position is missing.
        auto * ctx_dft = this->params.ctx_dft;
        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);
        if (pos_max < N - 2) {
            SPC_WRN("ctx_dft pos_max=%d < N-2=%d — process() did not run on every prefill ubatch. "
                    "Drafts may degrade.\n",
                    (int) pos_max, N - 2);
        }
    }

    bool process(const llama_batch & batch_in) override {
        if (batch_in.n_tokens <= 0) {
            return true;
        }

        if (batch_in.token == nullptr || batch_in.embd != nullptr) {
            return true;
        }

        const int32_t n_tokens = batch_in.n_tokens;

        // i_batch_beg[seq] / i_batch_end[seq]: inclusive batch indices of this seq's
        // first/last token in batch_in. Assumes per-seq tokens are contiguous within
        // the ubatch (server's default ordering).
        std::vector<int32_t> i_batch_beg(n_seq, -1);
        std::vector<int32_t> i_batch_end(n_seq, -1);
        for (int k = 0; k < n_tokens; ++k) {
            GGML_ASSERT(batch_in.n_seq_id[k] == 1);
            const llama_seq_id seq_id = batch_in.seq_id[k][0];
            if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
                continue;
            }
            i_batch_end[seq_id] = k;
            if (i_batch_beg[seq_id] < 0) {
                i_batch_beg[seq_id] = k;
            }
        }

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;

        // Interleave each extract_layer's hidden state into a contiguous buffer of
        // shape [n_tokens, target_layer_ids_n * n_embd_tgt]. Then run EAGLE3 encoder
        // to get one g_embd row per token.
        features_buf.resize((size_t) n_tokens * n_embd_enc, 0.0f);

        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            const float * layer = target_layer_ids[k] < n_layer_tgt
                ? llama_get_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k])
                : llama_get_embeddings_nextn(ctx_tgt);
            if (!layer) {
                GGML_ABORT("EAGLE3: target layer %d input not extracted.", target_layer_ids[k]);
            }
            for (int32_t i = 0; i < n_tokens; ++i) {
                float * dst = features_buf.data() + (size_t) i * n_embd_enc + k * (size_t) n_embd_tgt;
                const float * src = layer + (size_t) i * n_embd_tgt;
                std::memcpy(dst, src, (size_t) n_embd_tgt * sizeof(float));
            }
        }

        g_embd_buf.resize((size_t) n_tokens * n_embd_dec);

        // llama_encode() requires the full encoder batch to fit in n_ubatch.
        // Allow batch > ubatch: eagle3's per-token encoder can be chunked safely.
        const int32_t n_ubatch_dft = (int32_t) llama_n_ubatch(ctx_dft);
        for (int32_t i = 0; i < n_tokens; i += n_ubatch_dft) {
            const int32_t n_chunk = std::min(n_ubatch_dft, n_tokens - i);

            llama_batch enc_batch = {
                /*.n_tokens =*/ n_chunk,
                /*.token    =*/ nullptr,
                /*.embd     =*/ features_buf.data() + (size_t) i * n_embd_enc,
                /*.pos      =*/ nullptr,
                /*.n_seq_id =*/ nullptr,
                /*.seq_id   =*/ nullptr,
                /*.logits   =*/ nullptr,
            };
            const int32_t rc = llama_encode(ctx_dft, enc_batch);
            if (rc != 0) {
                SPC_ERR("llama_encode(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n",
                        rc, (int) n_chunk, (int) i);
                return false;
            }

            // g_embd has shape [n_chunk, n_embd_dec] in ctx_dft's pre-norm embeddings buffer.
            const float * g_embd_chunk = llama_get_embeddings_nextn(ctx_dft);
            GGML_ASSERT(g_embd_chunk && "EAGLE3 encoder produced no output.");
            std::memcpy(g_embd_buf.data() + (size_t) i * n_embd_dec,
                        g_embd_chunk,
                        (size_t) n_chunk * n_embd_dec * sizeof(float));
        }

        const float * g_embd = g_embd_buf.data();

        const size_t row_bytes = (size_t) n_embd_dec * sizeof(float);

        // EAGLE3 decoder input convention: at memory pos P the input pair is
        // (token[P+1], g_embd[P]). This shifts the token index "left by one" relative to g_embd.
        //
        // Per seq, in order:
        //   (a) cross-ubatch bridge — when applicable, write the previously-deferred
        //       pos using this ubatch's first token + pending_g_last.
        //   (b) main write loop — for k in [beg, end-1], write (token[k+1], g_embd[k])
        //       at pos[k]. The last training pos (k=end) is left unwritten = new
        //       deferred boundary, completed by the next process() or draft() call.
        //   (c) refresh deferred state — stash this ubatch's full g_embd into verify_g,
        //       update pending_g_last / pending_pos_last to the last row.
        common_batch_clear(batch);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            const int32_t beg = i_batch_beg[seq_id];
            const int32_t end = i_batch_end[seq_id];
            if (beg < 0 || end < 0) {
                continue;
            }

            // cross-ubatch bridge — complete the prior ubatch's deferred boundary.
            // Fires iff all three preconditions hold:
            //   1) pending_pos_last >= 0
            //   2) pending_pos_last + 1 == pos[beg]
            //   3) pending_pos_last > dft_pos_max // TODO: is this check needed?
            const llama_pos pending_pos = pending_pos_last[seq_id];
            if (pending_pos >= 0 && pending_pos + 1 == batch_in.pos[beg]) {
                const llama_pos dft_pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);
                if (pending_pos > dft_pos_max) {
                    common_batch_add(batch, batch_in.token[beg], pending_pos, { seq_id }, /*logits=*/ false);
                    std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec,
                                pending_g_last[seq_id].data(), row_bytes);
                }
            }

            for (int32_t k = beg; k < end; ++k) {
                common_batch_add(batch, batch_in.token[k + 1], batch_in.pos[k], { seq_id }, /*logits=*/ false);
                std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec,
                            g_embd + (size_t) k * n_embd_dec, row_bytes);
            }

            // refresh deferred state
            const int32_t n_rows = end - beg + 1;
            verify_pos_first[seq_id] = batch_in.pos[beg];
            pending_pos_last[seq_id] = batch_in.pos[end];
            verify_g_rows[seq_id]    = n_rows;
            verify_g[seq_id].resize((size_t) n_rows * n_embd_dec, 0.0f);
            std::memcpy(verify_g[seq_id].data(),       g_embd + (size_t) beg * n_embd_dec, row_bytes * n_rows);
            std::memcpy(pending_g_last[seq_id].data(), g_embd + (size_t) end * n_embd_dec, row_bytes);
        }

        if (batch.n_tokens > 0) {
            const int32_t rc = llama_decode(ctx_dft, batch);
            if (rc != 0) {
                SPC_ERR("llama_decode(ctx_dft) failed rc=%d (n_tokens=%d, ubatch_pos[0]=%d)\n",
                        rc, (int) batch.n_tokens, (int) batch_in.pos[0]);
                return false;
            }
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        const size_t row_bytes = (size_t) n_embd_dec * sizeof(float);

        // Complete the deferred boundary pair (dp.id_last, pending_g_last) at memory
        // pos pending_pos_last. dp.id_last is target's freshest sample (= corrected
        // token after verify, or first generated token after prefill), matching the
        // EAGLE3 input convention (token[P+1], g_embd[P]) at pos P.
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }
            if (pending_pos_last[seq_id] < 0) {
                continue;
            }

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, pending_pos_last[seq_id], -1);

            common_batch_add(batch, dp.id_last, pending_pos_last[seq_id], { seq_id }, true);
            std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec,
                        pending_g_last[seq_id].data(),
                        row_bytes);
        }

        if (batch.n_tokens == 0) {
            return;
        }

        int ret = llama_decode(ctx_dft, batch);
        if (ret != 0) {
            SPC_ERR("llama_decode returned %d\n", ret);
            return;
        }

        int i = 0;

        while (n_drafting > 0) {
            int i_batch = 0;

            common_batch_clear(batch);

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                common_sampler_sample(smpl, ctx_dft, i_batch, true);
                // pre-norm hidden state of this position becomes g_embd for the next step
                const float * prenorm = llama_get_embeddings_nextn_ith(ctx_dft, i_batch);
                ++i_batch;

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    SPC_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                const llama_token id = cur_p->data[0].id;

                // only collect very high-confidence draft tokens
                // (configurable via --spec-draft-p-min, set to 0.0 to disable early-stop)
                if (cur_p->data[0].p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);

                if (params.n_max <= (int) result.size()) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                common_batch_add(batch, id, pending_pos_last[seq_id] + (i + 1), { seq_id }, true);
                std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec, prenorm, row_bytes);
            }

            if (batch.n_tokens == 0) {
                break;
            }

            ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                SPC_ERR("llama_decode[%d] returned %d\n", i, ret);
                break;
            }

            ++i;
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool /*is_other*/) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        const int32_t n_rows = verify_g_rows[seq_id];
        if (n_rows <= 0) {
            return;
        }

        const int32_t i_g = std::min<int32_t>(n_accepted, n_rows - 1);
        pending_pos_last[seq_id] = verify_pos_first[seq_id] + i_g;
        std::memcpy(pending_g_last[seq_id].data(),
                    verify_g[seq_id].data() + (size_t) i_g * n_embd_dec,
                    (size_t) n_embd_dec * sizeof(float));
    }

    // we only need to stash the deferred boundary's g_embd row for recurrent/hybrid targets:
    // their single-position checkpoints drop it on restore
    bool need_boundary_stash() const {
        const llama_model * model_tgt = llama_get_model(params.ctx_tgt);
        return llama_model_is_recurrent(model_tgt) || llama_model_is_hybrid(model_tgt);
    }

    bool get_state(llama_seq_id seq_id, std::vector<uint8_t> & data) const override {
        if (!need_boundary_stash()) {
            return false;
        }
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq || pending_pos_last[seq_id] < 0) {
            return false;
        }

        const llama_pos          pos = pending_pos_last[seq_id];
        const std::vector<float> & g = pending_g_last[seq_id];

        data.resize(sizeof(llama_pos) + g.size() * sizeof(float));
        std::memcpy(data.data(),                     &pos,     sizeof(llama_pos));
        std::memcpy(data.data() + sizeof(llama_pos), g.data(), g.size() * sizeof(float));
        return true;
    }

    void set_state(llama_seq_id seq_id, const std::vector<uint8_t> & data) override {
        if (!need_boundary_stash()) {
            return;
        }
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }
        if (data.size() != sizeof(llama_pos) + (size_t) n_embd_dec * sizeof(float)) {
            return;
        }

        llama_pos pos = -1;
        std::memcpy(&pos, data.data(), sizeof(llama_pos));

        pending_pos_last[seq_id] = pos;
        pending_g_last[seq_id].resize(n_embd_dec);
        std::memcpy(pending_g_last[seq_id].data(), data.data() + sizeof(llama_pos), (size_t) n_embd_dec * sizeof(float));
    }
};

namespace spec {

struct dflash_breakdown_profiler {
    int64_t t_start_us    = 0;
    int64_t t_norm_us     = 0;
    int64_t t_output_us   = 0;
    int64_t dur_layers_us = 0;
    int64_t dur_head_us   = 0;
    int64_t dur_sel_us    = 0;
    int32_t last_n_tokens = 0;
    bool    active        = false;
    bool    deep_profile  = false;

    // Sub-layer profiling for up to 8 layers
    static constexpr int MAX_PROF_LAYERS = 8;
    int64_t t_layer_start_us   [MAX_PROF_LAYERS] = {0};
    int64_t dur_attn_us        [MAX_PROF_LAYERS] = {0};
    int64_t dur_ffn_us         [MAX_PROF_LAYERS] = {0};
    int64_t dur_layer_us       [MAX_PROF_LAYERS] = {0};
    int64_t last_checkpoint_us = 0;

    void reset(int32_t n_tokens) {
        t_start_us    = ggml_time_us();
        t_norm_us     = 0;
        t_output_us   = 0;
        dur_layers_us = 0;
        dur_head_us   = 0;
        dur_sel_us    = 0;
        last_n_tokens = n_tokens;
        last_checkpoint_us = t_start_us;
        for (int i = 0; i < MAX_PROF_LAYERS; ++i) {
            t_layer_start_us[i] = (i == 0) ? t_start_us : 0;
            dur_attn_us[i]      = 0;
            dur_ffn_us[i]       = 0;
            dur_layer_us[i]     = 0;
        }
    }

    static bool cb(struct ggml_tensor * t, bool ask, void * user_data) {
        auto * p = static_cast<dflash_breakdown_profiler *>(user_data);
        if (!p || !p->active) {
            return false;
        }
        if (ask) {
            if (std::strcmp(t->name, "result_norm") == 0)   return true;
            if (std::strcmp(t->name, "result_output") == 0) return true;
            if (p->deep_profile) {
                if (std::strncmp(t->name, "attn_conv_out-", 14) == 0) return true;
                if (std::strncmp(t->name, "ffn_out-", 8) == 0 && std::strstr(t->name, "(") == nullptr) return true;
                if (std::strncmp(t->name, "l_out-", 6) == 0)          return true;
            }
            return false;
        }
        const int64_t now = ggml_time_us();
        if (p->deep_profile) {
            // Only trace primary checkpoint nodes
            if (std::strstr(t->name, "(") == nullptr) {
                LOG_INF("[cb-trace] %-20s : +%6.2f ms (since last cp)\n", t->name, (now - p->last_checkpoint_us) / 1000.0f);
            }
        }
        if (std::strcmp(t->name, "result_norm") == 0) {
            p->t_norm_us     = now;
            p->dur_layers_us = now - p->t_start_us;
            p->last_checkpoint_us = now;
        } else if (std::strcmp(t->name, "result_output") == 0) {
            p->t_output_us = now;
            p->dur_head_us = now - p->t_norm_us;
            p->last_checkpoint_us = now;
        } else if (p->deep_profile) {
            if (std::strncmp(t->name, "attn_conv_out-", 14) == 0) {
                int il = std::atoi(t->name + 14);
                if (il >= 0 && il < MAX_PROF_LAYERS) {
                    p->dur_attn_us[il] = now - p->last_checkpoint_us;
                    p->last_checkpoint_us = now;
                }
            } else if (std::strncmp(t->name, "ffn_out-", 8) == 0 && std::strstr(t->name, "(") == nullptr) {
                int il = std::atoi(t->name + 8);
                if (il >= 0 && il < MAX_PROF_LAYERS) {
                    p->dur_ffn_us[il] = now - p->last_checkpoint_us;
                    p->last_checkpoint_us = now;
                }
            } else if (std::strncmp(t->name, "l_out-", 6) == 0) {
                int il = std::atoi(t->name + 6);
                if (il >= 0 && il < MAX_PROF_LAYERS) {
                    int64_t l_start = (il == 0) ? p->t_start_us : p->t_layer_start_us[il];
                    p->dur_layer_us[il] = now - l_start;
                    if (il + 1 < MAX_PROF_LAYERS) {
                        p->t_layer_start_us[il + 1] = now;
                    }
                    p->last_checkpoint_us = now;
                }
            }
        }
        return true;
    }
};

static dflash_breakdown_profiler g_dflash_prof;

struct pipe_job {
    llama_seq_id seq_id      = 0;
    llama_pos    pos0        = 0;
    int32_t      n_real      = 0;
    int32_t      k_exec      = 0;
    int32_t      k_hyp       = 0;
    llama_token  anchor      = -1;
    llama_pos    pos_anchor  = 0;
    bool         ingest_only = false;
};

struct pipe_result {
    bool         ingest_ok = false;
    bool         draft_ok  = false;
    llama_tokens tokens;
    bool         la_valid  = false;
    llama_token  la_token  = -1;
    float        draft_ms  = 0.0f;
    int64_t      t_work_us = 0;
};

class pipe_engine {
public:
    using worker_fn = std::function<void(const pipe_job & job, const std::atomic<bool> & cancel, pipe_result & res)>;

    enum class state : uint8_t {
        IDLE = 0,
        DECLARED,
        RUNNING,
        FINISHED
    };

    pipe_engine() = default;

    ~pipe_engine() {
        shutdown();
    }

    void start(worker_fn fn) {
        shutdown();
        {
            std::lock_guard<std::mutex> lock(mtx_);
            fn_        = std::move(fn);
            stop_      = false;
            cancel_    = false;
            st_        = state::IDLE;
            in_use_    = false;
            t_fire_us_ = 0;
        }
        worker_ = std::thread([this]() { worker_loop_(); });
    }

    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            if (!worker_.joinable()) {
                return;
            }
            stop_   = true;
            cancel_ = true;
            cv_work_.notify_all();
        }
        if (worker_.joinable()) {
            worker_.join();
        }
        std::lock_guard<std::mutex> lock(mtx_);
        st_     = state::IDLE;
        in_use_ = false;
        fn_     = nullptr;
    }

    bool in_use() const noexcept {
        return in_use_.load(std::memory_order_acquire);
    }

    bool declare(const pipe_job & job) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (!worker_.joinable() || in_use_.load(std::memory_order_relaxed) || st_ != state::IDLE) {
            return false;
        }
        job_       = job;
        res_       = pipe_result{};
        cancel_    = false;
        t_fire_us_ = 0;
        st_        = state::DECLARED;
        in_use_.store(true, std::memory_order_release);
        return true;
    }

    void launch() {
        std::lock_guard<std::mutex> lock(mtx_);
        if (st_ != state::DECLARED) {
            return;
        }
        t_fire_us_ = ggml_time_us();
        st_        = state::RUNNING;
        cv_work_.notify_one();
    }

    bool cancel_unlaunched() {
        std::lock_guard<std::mutex> lock(mtx_);
        if (st_ == state::DECLARED) {
            st_ = state::IDLE;
            in_use_.store(false, std::memory_order_release);
            return true;
        }
        return false;
    }

    void abort() {
        std::unique_lock<std::mutex> lock(mtx_);
        if (!in_use_.load(std::memory_order_relaxed)) {
            return;
        }
        cancel_.store(true, std::memory_order_release);
        if (st_ == state::DECLARED) {
            st_ = state::IDLE;
            in_use_.store(false, std::memory_order_release);
            return;
        }
        if (st_ == state::RUNNING) {
            cv_done_.wait(lock, [this]() { return st_ == state::FINISHED || stop_; });
        }
        st_ = state::IDLE;
        in_use_.store(false, std::memory_order_release);
    }

    bool join(int64_t & wait_us) {
        const int64_t t0 = ggml_time_us();
        std::unique_lock<std::mutex> lock(mtx_);
        if (st_ == state::DECLARED) {
            wait_us = 0;
            return false;
        }
        if (st_ == state::RUNNING) {
            cv_done_.wait(lock, [this]() { return st_ == state::FINISHED || stop_; });
        }
        wait_us = ggml_time_us() - t0;
        return st_ == state::FINISHED;
    }

    const pipe_result & result() const noexcept {
        return res_;
    }

    void release() {
        std::lock_guard<std::mutex> lock(mtx_);
        st_ = state::IDLE;
        in_use_.store(false, std::memory_order_release);
    }

    int64_t t_fire_us() const noexcept {
        std::lock_guard<std::mutex> lock(mtx_);
        return t_fire_us_;
    }

private:
    void worker_loop_() {
        while (true) {
            pipe_job job;
            worker_fn fn;
            {
                std::unique_lock<std::mutex> lock(mtx_);
                cv_work_.wait(lock, [this]() { return st_ == state::RUNNING || stop_; });
                if (stop_) {
                    break;
                }
                job = job_;
                fn  = fn_;
            }

            pipe_result res;
            const int64_t t_start = ggml_time_us();
            try {
                if (fn) {
                    fn(job, cancel_, res);
                }
            } catch (...) {
                res.ingest_ok = false;
                res.draft_ok  = false;
            }
            res.t_work_us = ggml_time_us() - t_start;

            {
                std::lock_guard<std::mutex> lock(mtx_);
                res_ = std::move(res);
                st_  = state::FINISHED;
                cv_done_.notify_all();
            }
        }
    }

    mutable std::mutex      mtx_;
    std::condition_variable cv_work_;
    std::condition_variable cv_done_;
    std::thread             worker_;
    worker_fn               fn_;
    pipe_job                job_;
    pipe_result             res_;
    std::atomic<bool>       cancel_{false};
    std::atomic<bool>       in_use_{false};
    bool                    stop_      = false;
    state                   st_        = state::IDLE;
    int64_t                 t_fire_us_ = 0;
};

} // namespace spec

// DFlash: block-diffusion drafting with a draft-side KV cache injection
struct common_speculative_impl_draft_dflash : public common_speculative_impl {
    common_params_speculative_draft params;

    llama_batch batch;        // noise tokens
    llama_batch batch_inject; // target features for KV cache injection

    std::vector<common_sampler_ptr> smpls;

    // backend sampler chain per seq, attached to ctx_dft
    std::vector<llama_sampler *> backend_chains;

    int32_t n_embd_dec = 0;  // draft context row width (n_embd_out: per-layer for DFly)
    int32_t n_embd_enc = 0;  // target_layer_ids_n * target_hidden_size
    int32_t n_embd_tgt = 0;  // target model hidden size

    int32_t     block_size    = 0;
    llama_token mask_token_id = 0;

    bool    is_dflash2     = false;
    bool    is_mrope       = false;
    int32_t selector_top_k = 0;

    // draft-dspark: the draft carries a Markov head. Comes from the model, not the
    // requested type. The DSpark path also truncates on confidence, and for
    // sample_from_anchor models it starts one row earlier.
    bool is_dspark = false;

    // dspark speculators
    bool sample_from_anchor = true;

    const int32_t * target_layer_ids   = nullptr; // model_dft's extract layer indices
    uint32_t        target_layer_ids_n = 0;

    // scratch buffer for concatenated target features [n_tokens, n_embd_enc]
    std::vector<float> features_buf;

    // Adaptive Speculative Arbiter (Sonnet Phase 3 Final):
    llama_token                id_think_start       = -1;
    llama_token                id_think_end         = -1;
    bool                       arbiter_active       = false; // false = shadow mode (fixed K=1), true = active mode (dynamic K in {0, 1, 3})
    int                        arbiter_log_interval = 32;    // telemetry logging interval in cycles
    std::vector<spec::arbiter> arbiters;
    std::vector<int32_t>       last_k_drafted;
    std::vector<int64_t>       cycle_count;
    std::vector<int64_t>       accept_count;
    std::vector<float>         last_draft_ms;

    // -----------------------------------------------------------------------------------------------------------
    // Stage S8: asynchronous worker pipeline (public entry points: common_speculative_pipeline_* in speculative-tap.h)
    //
    // Ownership rule: ctx_dft, batch, batch_inject, features_buf and smpls[] belong to exactly one thread at a time.
    // The main thread owns them except between engine.launch() (tap callback) and engine.join()/abort(); in that
    // window only pipe_work() may touch them. process() is the one main-thread path that could violate this and it
    // returns early while a worker job is in use. The worker never touches the arbiter.
    // -----------------------------------------------------------------------------------------------------------
    struct pipe_cycle {
        bool         valid       = false;  // an anchor guess exists for the batch being verified
        bool         job         = false;  // a worker job was declared (and, after disarm, launched) for it
        llama_seq_id seq_id      = 0;
        llama_pos    pos0        = 0;
        int32_t      n_real      = 0;
        int32_t      k_exec      = 0;
        int32_t      k_hyp       = 0;      // K the worker drafts for (hypothesis for the next cycle)
        llama_token  anchor      = -1;
        bool         ingest_only = false;
    };

    spec::layer_tap * tap              = nullptr;
    spec::pipe_engine engine;
    spec::pipe_stats  pstats;
    bool              pipe_attached    = false;
    bool              pipe_ingest_only = false;

    // lookahead anchor left by the last draft()/take(): the guess for id_last of the cycle after the batch about to be verified
    bool        la_valid = false;
    llama_token la_token = -1;
    int32_t     la_k     = 0;

    pipe_cycle  cyc;

    // worker draft handed from collect() to take()
    bool         take_ready    = false;
    int32_t      take_k        = 0;      // K the worker drafted for
    llama_token  take_id       = -1;     // actual bonus token == anchor (hit)
    llama_tokens take_tokens;
    bool         take_la_valid = false;
    llama_token  take_la_token = -1;
    float        take_draft_ms = 0.0f;

    common_speculative_impl_draft_dflash(const common_params_speculative & params, uint32_t n_seq,
            common_speculative_type type = COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH)
        : common_speculative_impl(type, n_seq)
        , params(params.draft)
    {
        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        GGML_ASSERT(ctx_tgt && ctx_dft && "DFlash requires ctx_tgt and ctx_dft to be set");

        const llama_model * model_dft = llama_get_model(ctx_dft);
        const llama_model * model_tgt = llama_get_model(ctx_tgt);

        target_layer_ids   = llama_model_target_layer_ids  (model_dft);
        target_layer_ids_n = llama_model_target_layer_ids_n(model_dft);
        GGML_ASSERT(target_layer_ids_n > 0 && "DFlash model has no target_layer_ids");

        // Both lineages declare general.architecture = dflash, so the requested type cannot
        // pick the draft path. The Markov head is the on-disk marker and is already loaded.
        is_dspark = llama_model_has_dspark_markov_head(model_dft);
        const bool type_says_dspark = (type == COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK);
        if (type_says_dspark != is_dspark) {
            LOG_WRN("%s: draft model carries %s, but --spec-type requested %s. Using %s, which is "
                    "what the model needs. The wrong path drops confidence truncation, and for "
                    "sample_from_anchor models it also reads the drafts one row late.\n", __func__,
                    is_dspark ? "a DSpark Markov head" : "no DSpark Markov head",
                    common_speculative_type_to_str(type).c_str(),
                    is_dspark ? "DSpark" : "DFlash");
        }

        n_embd_tgt    = llama_model_n_embd(model_tgt);
        n_embd_dec    = llama_model_n_embd_out(model_dft);
        n_embd_enc    = (int32_t) target_layer_ids_n * n_embd_tgt;

        // read the trained block size from the dflash.block_size metadata key
        block_size = 16;
        {
            char buf[32] = {};
            if (llama_model_meta_val_str(model_dft, "dflash.block_size", buf, sizeof(buf)) >= 0) {
                block_size = std::atoi(buf);
            }
            if (llama_model_meta_val_str(model_dft, "dflash.sample_from_anchor", buf, sizeof(buf)) >= 0) {
                sample_from_anchor = std::strcmp(buf, "true") == 0;
            }
        }

        selector_top_k = llama_model_dflash_selector_top_k(model_dft);
        is_dflash2     = selector_top_k > 0;
        mask_token_id = llama_vocab_mask(llama_model_get_vocab(model_dft));

        // without a mask token every masked slot is drafted as token id -1: runs, accepts nothing,
        // and reads as a bad drafter instead of a bad conversion that dropped the vocab
        GGML_ASSERT(mask_token_id != LLAMA_TOKEN_NULL &&
                    "draft model has no mask token: check tokenizer.ggml.mask_token_id and tokenizer.ggml.model (a 'none' stub skips the vocab) in the GGUF");

        // Initialize Adaptive Speculative Arbiter (Phase 4 Specification)
        const char * env_mode = std::getenv("LLAMA_SPEC_ARBITER_MODE");
        std::string arbiter_mode_str = env_mode ? env_mode : "shadow";

        spec::arbiter_mode op_mode = spec::arbiter_mode::SHADOW;
        if (arbiter_mode_str == "active") op_mode = spec::arbiter_mode::ACTIVE;
        else if (arbiter_mode_str == "auto") op_mode = spec::arbiter_mode::AUTO;

        const char * env_interval = std::getenv("LLAMA_SPEC_ARBITER_LOG_INTERVAL");
        if (env_interval) {
            arbiter_log_interval = std::max(1, std::atoi(env_interval));
        }

        const char * env_drafter = std::getenv("LLAMA_SPEC_DRAFTER");
        spec::drafter_kind dkind = spec::drafter_kind::BLOCK;
        if (env_drafter && std::string(env_drafter) == "ar") {
            dkind = spec::drafter_kind::AUTOREGRESSIVE;
        }

        const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
        auto tok_s = common_tokenize(vocab_tgt, "<think>", false, true);
        if (!tok_s.empty()) id_think_start = tok_s[0];
        auto tok_e = common_tokenize(vocab_tgt, "</think>", false, true);
        if (!tok_e.empty()) id_think_end = tok_e[0];

        spec::arbiter_params aparams;
        aparams.op_mode  = op_mode;
        aparams.drafter  = dkind;
        aparams.id_open  = id_think_start;
        aparams.id_close = id_think_end;
        aparams.allow_k2 = false; // K=2 is dominated on M3
        aparams.force_k0_in_think = false; // let the mathematical model decide

        const char * env_pipe = std::getenv("LLAMA_SPEC_PIPELINE");
        if (env_pipe && (std::strcmp(env_pipe, "1") == 0 || std::strcmp(env_pipe, "true") == 0)) {
            aparams.pipelined = true;
        }

        const char * env_pmode = std::getenv("LLAMA_SPEC_PIPE_MODE");
        if (env_pmode && (std::strcmp(env_pmode, "ingest") == 0 || std::strcmp(env_pmode, "b0") == 0)) {
            pipe_ingest_only = true;
            LOG_INF("%s: Stage S8 Path B0 (ingest-only pipeline) enabled\n", __func__);
        }

        arbiters.clear();
        for (uint32_t s = 0; s < n_seq; ++s) {
            arbiters.emplace_back(aparams);
        }
        last_k_drafted.assign(n_seq, 1);
        cycle_count.assign(n_seq, 0);
        accept_count.assign(n_seq, 0);
        last_draft_ms.assign(n_seq, 0.0f);

        LOG_INF("%s: speculative arbiter initialized (mode=%s, drafter=%s, think_start=%d, think_end=%d, log_interval=%d, n_seq=%u)\n",
                __func__, arbiter_mode_str.c_str(), dkind == spec::drafter_kind::BLOCK ? "block" : "ar",
                id_think_start, id_think_end, arbiter_log_interval, n_seq);

        LOG_INF("%s: adding speculative implementation '%s'\n", __func__, common_speculative_type_to_str(type).c_str());
        LOG_INF("%s: - n_max=%d, n_min=%d, p_min=%.2f\n", __func__, this->params.n_max, this->params.n_min, this->params.p_min);
        LOG_INF("%s: - block_size=%d, mask_token_id=%d, n_extract=%u, sample_from_anchor=%s, lineage=%s\n", __func__,
                block_size, mask_token_id, target_layer_ids_n, sample_from_anchor ? "true" : "false",
                is_dspark ? "dspark" : "dflash");

        // DFlash input is [id_last, <mask> * (block_size-1)]: in-place denoising yields at most
        // block_size-1 draft tokens, anchor-first DSpark yields a full block_size draft tokens
        const int32_t n_draft_max = is_dspark && sample_from_anchor ? block_size : block_size - 1;
        if (this->params.n_max > n_draft_max || this->params.n_min > n_draft_max) {
            LOG_WRN("%s: requested draft size (n_max=%d, n_min=%d) exceeds the trained block size %d -- clamping to %d\n",
                    __func__, this->params.n_max, this->params.n_min, block_size, n_draft_max);
            this->params.n_max = std::min(this->params.n_max, n_draft_max);
            this->params.n_min = std::min(this->params.n_min, n_draft_max);
        }

        batch        = llama_batch_init(llama_n_batch(ctx_dft), 0,          n_seq);
        batch_inject = llama_batch_init(llama_n_batch(ctx_dft), n_embd_dec, n_seq);

        // embd batches on an M-RoPE draft need 4 position rows per token
        is_mrope = llama_model_rope_type(model_dft) == LLAMA_ROPE_TYPE_MROPE;
        if (is_mrope) {
            free(batch_inject.pos);
            batch_inject.pos = (llama_pos *) malloc(sizeof(llama_pos) * 4 * llama_n_batch(ctx_dft));
        }

        smpls.resize(n_seq);
        for (auto & s : smpls) {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            sparams.top_k    = 10;
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            s.reset(common_sampler_init(model_dft, sparams));
        }

        // offload draft sampling to the backend
        backend_chains.assign(n_seq, nullptr);
        if (this->params.backend_sampling && !is_dflash2) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(10));

                if (!llama_set_sampler(ctx_dft, seq_id, chain)) {
                    SPC_WRN("backend offload failed for seq_id=%d; using CPU sampler\n", (int) seq_id);
                    llama_sampler_free(chain);
                    chain = nullptr;
                }
                backend_chains[seq_id] = chain;
            }
        }

        // turn on extraction of the target layers' input embeddings
        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            llama_set_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k], true);
        }

        // DFlash2 reads its selector lattice from h_nextn and never consumes raw logits.
        llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ !is_dflash2);
        llama_set_causal_attn(ctx_dft, false); // DFlash needs non-causal attention
    }

    ~common_speculative_impl_draft_dflash() override {
        engine.shutdown(); // the worker owns ctx_dft while a job runs: join it first

        auto * ctx_dft = this->params.ctx_dft;
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) backend_chains.size(); ++seq_id) {
            if (backend_chains[seq_id] == nullptr) {
                continue;
            }
            if (ctx_dft) {
                llama_set_sampler(ctx_dft, seq_id, nullptr);
            }
            llama_sampler_free(backend_chains[seq_id]);
        }
        backend_chains.clear();

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) arbiters.size(); ++seq_id) {
            LOG_INF("[arbiter-audit seq=%d] total_draft_cycles=%lld total_arbiter_updates=%lld (1:1 lockstep: %s)\n",
                    (int) seq_id, (long long) cycle_count[seq_id], (long long) accept_count[seq_id],
                    cycle_count[seq_id] == accept_count[seq_id] ? "VERIFIED" : "MISMATCH");
        }

        llama_batch_free(batch);
        llama_batch_free(batch_inject);
    }

    // ---- Stage S8 pipeline internals -----------------------------------------------------------------------------

    // Encode + inject `n_rows` rows of target features into the draft KV cache.
    //   layer_rows[k] : row 0 of extract layer k, [n_rows, n_embd_tgt] f32, row stride n_embd_tgt
    //   pos[i]        : target position of row i
    // Shared by process() (rows from llama_get_embeddings_layer_inp) and the worker (rows from the tap).
    bool ingest_rows(llama_seq_id seq_id, int32_t n_rows, const std::vector<const float *> & layer_rows, const llama_pos * pos) {
        auto * ctx_dft = this->params.ctx_dft;

        const int32_t n_ubatch = (int32_t) llama_n_ubatch(ctx_dft);

        for (int32_t offset = 0; offset < n_rows; offset += n_ubatch) {
            const int32_t n_chunk = std::min(n_ubatch, n_rows - offset);

            // gather this chunk's target features, interleaved by extract layer
            features_buf.resize((size_t) n_chunk * n_embd_enc);
            for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
                for (int32_t i = 0; i < n_chunk; ++i) {
                    float       * dst = features_buf.data() + (size_t) i * n_embd_enc + k * (size_t) n_embd_tgt;
                    const float * src = layer_rows[k] + (size_t) (offset + i) * n_embd_tgt;
                    std::memcpy(dst, src, (size_t) n_embd_tgt * sizeof(float));
                }
            }

            // fuse extracted features through DFlash encoder
            // M-RoPE drafts read 4 position rows per token from embd batches, so pass them explicitly
            std::vector<llama_pos> enc_pos;
            if (is_mrope) {
                enc_pos.resize((size_t) 4 * n_chunk);
                for (int32_t i = 0; i < n_chunk; ++i) {
                    const llama_pos p = pos[offset + i];
                    enc_pos[0 * n_chunk + i] = p;
                    enc_pos[1 * n_chunk + i] = p;
                    enc_pos[2 * n_chunk + i] = p;
                    enc_pos[3 * n_chunk + i] = 0;
                }
            }

            llama_batch enc_batch = {
                /*.n_tokens =*/ n_chunk,
                /*.token    =*/ nullptr,
                /*.embd     =*/ features_buf.data(),
                /*.pos      =*/ is_mrope ? enc_pos.data() : nullptr,
                /*.n_seq_id =*/ nullptr,
                /*.seq_id   =*/ nullptr,
                /*.logits   =*/ nullptr,
            };

            int32_t rc = llama_encode(ctx_dft, enc_batch);
            if (rc != 0) {
                LOG_ERR("%s: llama_encode(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n",
                        __func__, rc, (int) n_chunk, (int) offset);
                return false;
            }

            const float * inp_g = llama_get_embeddings_nextn(ctx_dft);
            GGML_ASSERT(inp_g && "DFlash encoder produced no output.");

            // inject the DFlash decoder K/V cache at the tokens' target positions
            batch_inject.n_tokens = n_chunk;
            std::memcpy(batch_inject.embd, inp_g, (size_t) n_chunk * n_embd_dec * sizeof(float));

            for (int32_t i = 0; i < n_chunk; ++i) {
                const llama_pos p = pos[offset + i];
                batch_inject.pos[i] = p;
                if (is_mrope) {
                    batch_inject.pos[1 * n_chunk + i] = p;
                    batch_inject.pos[2 * n_chunk + i] = p;
                    batch_inject.pos[3 * n_chunk + i] = 0;
                }
                batch_inject.n_seq_id[i]  = 1;
                batch_inject.seq_id[i][0] = seq_id;
                batch_inject.logits[i]    = false;
            }
            rc = llama_decode(ctx_dft, batch_inject);
            if (rc != 0) {
                LOG_ERR("%s: llama_decode(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n",
                        __func__, rc, (int) n_chunk, (int) offset);
                return false;
            }
        }

        return true;
    }

    // arbiter telemetry line, shared by draft() and take()
    void log_arbiter(llama_seq_id seq_id, int k_pred, int k_actual) {
        if (cycle_count[seq_id] % arbiter_log_interval == 0) {
            const auto snap = arbiters[seq_id].snapshot(arbiters[seq_id].in_think() ? spec::mode::think : spec::mode::content);
            LOG_INF("[spec-arbiter seq=%d cyc=%lld phase=%s mode=%s%s] K_pred=%d K_act=%d | td0=%.1fms td=%.2fms | slack13=%.3f lcb=%.3f beta=%.2f | c=(%.3f, %.3f, %.3f) | TPS K0..3=[%.2f, %.2f, %.2f, %.2f]\n",
                    (int) seq_id, (long long) cycle_count[seq_id],
                    arbiters[seq_id].in_think() ? "THINK" : "CONTENT",
                    snap.is_active ? "active" : "shadow",
                    arbiters[seq_id].get_mode() == spec::arbiter_mode::AUTO ? (snap.is_active ? "(promoted)" : "(evaluating)") : "",
                    k_pred, k_actual,
                    snap.t_d0, snap.t_d,
                    snap.slack13, snap.lcb, snap.beta_t,
                    snap.c[0], snap.c[1], snap.c[2],
                    snap.tps[0], snap.tps[1], snap.tps[2], snap.tps[3]);
        }
    }

    // Reads the drafted block that llama_decode(ctx_dft) just produced for one sequence into `result`.
    // `beg` is the batch row of the block's first token. Moved verbatim out of draft() so the worker can reuse it.
    void read_block_result(llama_seq_id seq_id, int32_t beg, int32_t n_block_tokens, llama_tokens & result) {
        auto & ctx_dft = params.ctx_dft;
        auto * smpl    = smpls[seq_id].get();

        if (is_dflash2) {
            const float * lattice = llama_get_embeddings_nextn(ctx_dft);
            GGML_ASSERT(lattice && "DFlash2 selector produced no lattice");

            int32_t predecessor = 0;
            for (int32_t i = 1; i < n_block_tokens; ++i) {
                const float * row = lattice + (size_t) (beg + i) * n_embd_dec;
                const float * scores = row + selector_top_k + (size_t) predecessor * selector_top_k;

                predecessor = (int32_t) std::distance(scores,
                        std::max_element(scores, scores + selector_top_k));
                if (params.p_min > 0.0f) {
                    // softmax(scores) at the argmax, i.e. 1 / sum(exp(s_k - s_max))
                    float sum = 0.0f;
                    for (int32_t k = 0; k < selector_top_k; ++k) {
                        sum += std::exp(scores[k] - scores[predecessor]);
                    }
                    if (1.0f / sum < params.p_min) {
                        break;
                    }
                }
                result.push_back((llama_token) row[predecessor]);
            }

            if (result.size() < (size_t) params.n_min) {
                result.clear();
            }
            return;
        }

        if (is_dspark) {
            // DSpark: read from the first draft slot, truncate below the confidence threshold
            const float * conf = params.p_min > 0.0f ? llama_get_embeddings_nextn(ctx_dft) : nullptr;
            // bonus-anchor drafts read the mask positions only, like DFlash
            const int32_t i_draft_beg = sample_from_anchor ? 0 : 1;
            for (int32_t i = i_draft_beg; i < n_block_tokens; ++i) {
                const int32_t idx = beg + i;

                if (conf && conf[(size_t) idx * n_embd_dec] < params.p_min) {
                    break;
                }

                common_sampler_sample(smpl, ctx_dft, idx, true);

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    LOG_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                const llama_token id = cur_p->data[0].id;

                common_sampler_accept(smpl, id, true);

                result.push_back(id);
            }
        } else {
            // greedily read the predicted block at this sequence's noise positions 1..n_block_tokens-1
            for (int32_t i = 1; i < n_block_tokens; ++i) {
                common_sampler_sample(smpl, ctx_dft, beg + i, true);

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    LOG_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i - 1, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                const llama_token id = cur_p->data[0].id;

                if (cur_p->data[0].p < params.p_min) {
                    break;
                }

                common_sampler_accept(smpl, id, true);

                result.push_back(id);
            }
        }

        if (result.size() < (size_t) params.n_min) {
            result.clear();
        }

    }

    // Single-sequence noise-block draft for the worker. Same batch layout as draft(); touches ctx_dft, batch and
    // smpls[seq_id] only, never the arbiter. Leaves no noise rows in the draft KV cache.
    bool draft_block(llama_seq_id seq_id, llama_token id_anchor, llama_pos pos_anchor, int32_t n_draft,
            llama_tokens & result, float & total_ms) {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);
        common_sampler_reset(smpls[seq_id].get());

        const int32_t n_block_tokens = n_draft + (is_dspark && sample_from_anchor ? 0 : 1);
        for (int32_t i = 0; i < n_block_tokens; ++i) {
            common_batch_add(batch, i == 0 ? id_anchor : mask_token_id, pos_anchor + i, { seq_id }, !is_dflash2);
        }

        const int64_t t0 = ggml_time_us();
        const int ret = llama_decode(ctx_dft, batch);
        const int64_t t1 = ggml_time_us();
        llama_synchronize(ctx_dft);
        const int64_t t2 = ggml_time_us();
        total_ms = (t2 - t0) / 1000.0f;
        const float enq_ms  = (t1 - t0) / 1000.0f;
        const float sync_ms = (t2 - t1) / 1000.0f;
        LOG_INF("[dflash-worker-timer] batch_tokens=%d total_td=%.2f ms (cpu_enq=%.2f ms, gpu_sync=%.2f ms)\n",
                (int) batch.n_tokens, total_ms, enq_ms, sync_ms);
        if (ret != 0) {
            LOG_WRN("%s: llama_decode(ctx_dft) returned %d\n", __func__, ret);
            llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, pos_anchor, -1);
            return false;
        }

        read_block_result(seq_id, 0, n_block_tokens, result);

        // the block rows are scratch: drop them so the draft KV cache holds only target-feature rows
        return llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, pos_anchor, -1);
    }

    bool pipe_lookahead_allowed() const {
        if (pipe_ingest_only) {
            return false;
        }
        const char * env = std::getenv("LLAMA_SPEC_LOOKAHEAD");
        return env ? (std::atoi(env) != 0) : true;
    }

    // K the arbiter would pick right now, without side effects (hypothesis for cycle N+1)
    int32_t pipe_k_peek(llama_seq_id seq_id) const {
        if (const char * env_force_k = std::getenv("LLAMA_SPEC_FORCE_K")) {
            return std::atoi(env_force_k);
        }
        return arbiters[seq_id].select_k();   // const overload
    }

    // true when the worker already ingested the rows of this verify batch (so process() must not touch ctx_dft)
    bool pipe_owns_ingest(const llama_batch & b) const {
        if (!pipe_attached || !cyc.job) {
            return false;
        }
        if (b.n_tokens != cyc.n_real || b.pos == nullptr || b.pos[0] != cyc.pos0) {
            return false;
        }
        return engine.in_use();
    }

    // worker body: runs on the worker thread, owns ctx_dft for its duration
    void pipe_work(const spec::pipe_job & job, const std::atomic<bool> & cancel, spec::pipe_result & res) {
        const int64_t t0 = ggml_time_us();

        // 1. ingest rows [0, n_real) straight from the tap host copies (no second pass through process())
        std::vector<const float *> rows(target_layer_ids_n);
        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            rows[k] = tap->rows(k);
        }
        std::vector<llama_pos> pos((size_t) job.n_real);
        for (int32_t i = 0; i < job.n_real; ++i) {
            pos[i] = job.pos0 + i;
        }
        res.ingest_ok = ingest_rows(job.seq_id, job.n_real, rows, pos.data());
        if (!res.ingest_ok || cancel.load() || job.ingest_only) {
            res.t_work_us = ggml_time_us() - t0;
            return;
        }

        // 2. hypothesis draft for cycle N+1: [anchor, <mask> * ...] at pos_anchor.. ; the extra (lookahead) row carries
        //    the anchor guess for cycle N+2, its cost is hidden behind the remaining target layers
        const int32_t k_hyp   = job.k_hyp;
        const bool    want_la = pipe_lookahead_allowed() && k_hyp + 1 <= (int32_t) params.n_max;
        const int32_t n_draft = want_la ? k_hyp + 1 : k_hyp;

        llama_tokens toks;
        res.draft_ok = draft_block(job.seq_id, job.anchor, job.pos_anchor, n_draft, toks, res.draft_ms);

        if (want_la && toks.size() == (size_t) k_hyp + 1) {
            res.la_token = toks.back();
            res.la_valid = true;
            toks.pop_back();
        }
        if (toks.size() < (size_t) params.n_min) {
            toks.clear();
            res.la_valid = false;
        }
        res.tokens = std::move(toks);
    }

    // bit-exact check of the tap host copies against what llama_decode published for the same batch
    bool pipe_features_match(int32_t n_real) const {
        if (tap->n_embd() != n_embd_tgt) {
            return false;
        }
        for (size_t k = 0; k < tap->n_slots(); ++k) {
            const float * direct = llama_get_embeddings_layer_inp(params.ctx_tgt, tap->slot_layer(k));
            if (direct == nullptr) {
                return false;
            }
            if (std::memcmp(tap->rows(k), direct, (size_t) n_real * (size_t) n_embd_tgt * sizeof(float)) != 0) {
                return false;
            }
        }
        return true;
    }

    // the worker ingested bad/partial rows: drop them and ingest the rows llama_decode published (sequential semantics)
    bool pipe_reingest(const pipe_cycle & c) {
        auto * ctx_dft = params.ctx_dft;
        auto * ctx_tgt = params.ctx_tgt;
        llama_memory_seq_rm(llama_get_memory(ctx_dft), c.seq_id, c.pos0, -1);

        std::vector<const float *> rows(target_layer_ids_n);
        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            rows[k] = llama_get_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k]);
            if (rows[k] == nullptr) {
                return false;
            }
        }
        std::vector<llama_pos> pos((size_t) c.n_real);
        for (int32_t i = 0; i < c.n_real; ++i) {
            pos[i] = c.pos0 + i;
        }
        return ingest_rows(c.seq_id, c.n_real, rows, pos.data());
    }

    bool pipe_attach(spec::layer_tap * t) {
        if (t == nullptr || pipe_attached) {
            return pipe_attached;
        }
        if (n_seq != 1 || arbiters.size() != 1 || !arbiters[0].is_pipelined()) {
            return false;
        }
        const auto rm_dft = common_context_can_seq_rm(params.ctx_dft);
        if (rm_dft == COMMON_CONTEXT_SEQ_RM_TYPE_NO || rm_dft == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
            rm_dft == COMMON_CONTEXT_SEQ_RM_TYPE_RS) {
            LOG_WRN("%s: draft context cannot drop a tail of positions: pipeline disabled\n", __func__);
            return false;
        }
        t->configure(target_layer_ids, target_layer_ids_n, n_embd_tgt);
        if (!t->configured() || t->n_slots() != target_layer_ids_n || t->n_embd() != n_embd_tgt) {
            return false;
        }
        tap = t;
        engine.start([this](const spec::pipe_job & job, const std::atomic<bool> & cancel, spec::pipe_result & res) {
            pipe_work(job, cancel, res);
        });
        pipe_attached = true;
        LOG_INF("%s: DFlash worker pipeline attached (n_slots=%zu, n_embd=%d)\n", __func__, t->n_slots(), (int) n_embd_tgt);
        return true;
    }

    bool pipe_arm(llama_seq_id seq_id, llama_pos pos0, int32_t n_real, int32_t k_exec) {
        if (!pipe_attached || seq_id != 0) {
            la_valid = false;
            return false;
        }
        if (engine.in_use()) {
            engine.abort();   // defensive: the caller skipped collect()
        }
        cyc = pipe_cycle{};

        if (pipe_ingest_only) {
            if (k_exec < 1 || n_real != k_exec + 1 || pstats.disabled || !tap->configured()) {
                return false;
            }
            cyc.valid       = true;
            cyc.seq_id      = seq_id;
            cyc.pos0        = pos0;
            cyc.n_real      = n_real;
            cyc.k_exec      = k_exec;
            cyc.ingest_only = true;

            spec::pipe_job job;
            job.seq_id      = seq_id;
            job.pos0        = pos0;
            job.n_real      = n_real;
            job.k_exec      = k_exec;
            job.ingest_only = true;
            if (!engine.declare(job)) {
                return false;
            }
            cyc.job = true;
            tap->arm(n_real, [this]() { engine.launch(); });
            return true;
        }

        // the anchor must belong to exactly this batch
        const bool anchor_ok = la_valid && la_k == k_exec && k_exec >= 1 && n_real == k_exec + 1;
        la_valid = false;
        if (!anchor_ok) {
            return false;
        }
        cyc.valid  = true;
        cyc.seq_id = seq_id;
        cyc.pos0   = pos0;
        cyc.n_real = n_real;
        cyc.k_exec = k_exec;
        cyc.anchor = la_token;   // still holds the value draft()/take() stored

        if (pstats.disabled || !tap->configured()) {
            return false;        // anchor is still tracked so the arbiter keeps learning g
        }

        const int32_t k_hyp = std::min(pipe_k_peek(seq_id), (int32_t) params.n_max);
        if (k_hyp < 1) {
            return false;
        }

        cyc.k_hyp = k_hyp;

        spec::pipe_job job;
        job.seq_id     = seq_id;
        job.pos0       = pos0;
        job.n_real     = n_real;
        job.k_exec     = k_exec;
        job.k_hyp      = k_hyp;
        job.anchor     = cyc.anchor;
        job.pos_anchor = pos0 + n_real;
        if (!engine.declare(job)) {
            return false;
        }
        cyc.job = true;
        tap->arm(n_real, [this]() { engine.launch(); });
        return true;
    }

    void pipe_disarm(int64_t t_decode_end_us) {
        if (tap == nullptr) {
            return;
        }
        if (cyc.job) {
            if (engine.cancel_unlaunched()) {
                // armed, but the boundary node never fired or the capture was rejected: nothing was handed over
                pstats.tap_fail++;
                cyc.job = false;
            } else {
                pstats.submitted++;
                const int64_t t_fire = engine.t_fire_us();
                if (t_fire > 0 && t_decode_end_us > t_fire) {
                    pstats.t_window_us += t_decode_end_us - t_fire;
                }
            }
        }
        tap->disarm();
    }

    void pipe_collect(llama_seq_id seq_id, int n_accepted, llama_token id_last_actual) {
        // whatever collect() produced for the previous cycle is stale now
        take_ready = false;
        take_tokens.clear();

        if (!pipe_attached || !cyc.valid) {
            return;
        }
        const pipe_cycle c = cyc;
        cyc = pipe_cycle{};

        const bool all_accepted = n_accepted == c.k_exec;
        const bool anchor_match = all_accepted && id_last_actual == c.anchor;
        if (!c.ingest_only && all_accepted) {
            arbiters[seq_id].report_anchor(anchor_match);   // g = P(anchor == bonus | all K accepted)
        }
        if (!c.job) {
            return;
        }

        // join first: ctx_dft goes back to the main thread on every outcome
        int64_t wait_us = 0;
        const bool have = engine.join(wait_us);
        pstats.t_wait_us += wait_us;
        if (!have) {
            return;
        }
        const spec::pipe_result & r = engine.result();
        pstats.t_worker_us += r.t_work_us;

        // the worker ingested rows from the tap; they are shared by every outcome, so verify them always
        if (!r.ingest_ok || !pipe_features_match(c.n_real)) {
            pstats.verify_fail++;
            pstats.disabled = true;
            LOG_ERR("%s: tap features rejected (ingest_ok=%d); pipeline disabled, re-ingesting sequentially\n",
                    __func__, (int) r.ingest_ok);
            if (!pipe_reingest(c)) {
                LOG_ERR("%s: sequential re-ingest failed; draft quality will degrade\n", __func__);
            }
            engine.release();
            return;
        }

        if (c.ingest_only) {
            pstats.ingest_jobs++;
            engine.release();
            return;
        }

        if (!all_accepted) {
            pstats.miss_accept++;
        } else if (!anchor_match) {
            pstats.miss_anchor++;
        } else if (!r.draft_ok || r.tokens.empty()) {
            pstats.miss_empty++;
        } else {
            take_ready    = true;
            take_k        = c.k_hyp;           // the K the worker drafted for; take() compares it with the arbiter's K
            take_id       = id_last_actual;    // == anchor
            take_tokens   = r.tokens;
            take_la_valid = r.la_valid;
            take_la_token = r.la_token;
            take_draft_ms = r.draft_ms;
        }
        engine.release();
    }

    bool pipe_take(llama_seq_id seq_id, int n_draft_max, llama_tokens & out) {
        if (!pipe_attached || !take_ready || seq_id != 0) {
            return false;
        }
        take_ready = false;
        if (pstats.disabled || take_tokens.empty()) {
            take_tokens.clear();
            return false;
        }

        auto & arb = arbiters[seq_id];
        arb.feed_token_id(take_id);

        const int k_pred   = arb.pred_k();
        int       k_actual = arb.select_k();
        if (const char * env_force_k = std::getenv("LLAMA_SPEC_FORCE_K")) {
            k_actual = std::atoi(env_force_k);
        }
        const int32_t k_eff = std::min((int32_t) k_actual, (int32_t) params.n_max);

        // the sequential path decides these cases itself (it re-reads the arbiter and the generation budget)
        if (k_eff != take_k || n_draft_max < (int) take_tokens.size()) {
            pstats.miss_k++;
            take_tokens.clear();
            return false;
        }

        cycle_count[seq_id]++;
        last_k_drafted[seq_id] = is_dflash2 ? (int32_t) k_actual : (int32_t) take_tokens.size();
        last_draft_ms [seq_id] = take_draft_ms;
        log_arbiter(seq_id, k_pred, k_actual);

        // the worker's lookahead row is the anchor guess for the batch we are about to verify
        la_valid = take_la_valid;
        la_token = take_la_token;
        la_k     = (int32_t) take_tokens.size();

        out = std::move(take_tokens);
        take_tokens.clear();
        pstats.hits++;
        return true;
    }

    // drop any in-flight / ready hypothesis (error paths, checkpoint restores); ctx_dft is back with the caller on return
    void pipe_abort() {
        if (pipe_attached) {
            engine.abort();
        }
        cyc = pipe_cycle{};
        take_ready = false;
        take_tokens.clear();
        la_valid = false;
        if (tap != nullptr) {
            tap->disarm();
        }
    }


    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        if (pipe_attached) {
            pipe_abort();
        }

        if (seq_id < (llama_seq_id) arbiters.size()) {
            arbiters[seq_id].reset(false);
            for (auto t : prompt) {
                arbiters[seq_id].feed_token_id(t);
            }
            last_k_drafted[seq_id] = 1;
            cycle_count[seq_id] = 0;
            last_draft_ms[seq_id] = 0.0f;
        }

        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }

        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(params.ctx_dft), seq_id);
        if (pos_max < N - 1) {
            LOG_WRN("%s: ctx_dft pos_max=%d < N-1=%d - process() did not run on every prefill ubatch. "
                    "Drafts may degrade.\n",
                    __func__, (int) pos_max, N - 1);
        }
    }

    bool process(const llama_batch & batch_in) override {
        if (batch_in.n_tokens <= 0) {
            return true;
        }

        // Target prefill may contain token IDs or multimodal embeddings. Both
        // produce the target-layer features used to seed the draft KV cache, so
        // skipping the embedding batches leaves a hole in the draft's cache and
        // the next injection fails to initialize.
        // TODO: revisit after https://github.com/ggml-org/llama.cpp/pull/24669 is merged
        const bool has_tokens     = batch_in.token != nullptr;
        const bool has_embeddings = batch_in.embd  != nullptr;
        if (has_tokens == has_embeddings) {
            return true;
        }

        const int32_t n_tokens = batch_in.n_tokens;

        // Stage S8: the worker ingested exactly these rows from the tap and owns ctx_dft until collect() joins it
        if (pipe_owns_ingest(batch_in)) {
            return true;
        }

        // per-seq inclusive batch range (assumes each seq's tokens are contiguous in the batch)
        std::vector<int32_t> i_batch_beg(n_seq, -1);
        std::vector<int32_t> i_batch_end(n_seq, -1);
        for (int32_t k = 0; k < n_tokens; ++k) {
            GGML_ASSERT(batch_in.n_seq_id[k] == 1);
            const llama_seq_id seq_id = batch_in.seq_id[k][0];
            if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
                continue;
            }
            i_batch_end[seq_id] = k;
            if (i_batch_beg[seq_id] < 0) {
                i_batch_beg[seq_id] = k;
            }
        }

        auto * ctx_tgt = this->params.ctx_tgt;

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_batch_beg[seq_id] < 0) {
                continue;
            }
            const int32_t n_rows = i_batch_end[seq_id] - i_batch_beg[seq_id] + 1;

            std::vector<const float *> layer_rows(target_layer_ids_n);
            for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
                const float * layer = llama_get_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k]);
                if (!layer) {
                    GGML_ABORT("DFlash: target layer %d input not extracted.", target_layer_ids[k]);
                }
                layer_rows[k] = layer + (size_t) i_batch_beg[seq_id] * n_embd_tgt;
            }

            if (!ingest_rows(seq_id, n_rows, layer_rows, batch_in.pos + i_batch_beg[seq_id])) {
                return false;
            }
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // build one batch holding every drafting sequence's noise block into a single decode)
        // record where each block starts and its size
        std::vector<int32_t> i_block_beg(n_seq, -1);
        std::vector<int32_t> n_block    (n_seq,  0);
        std::vector<uint8_t> la_requested(n_seq, 0);   // lookahead row appended to this seq's block
        std::vector<int32_t> k_eff_seq   (n_seq, 0);

        la_valid = false;                              // a stale anchor must never survive a draft

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            // Adaptive Speculative Arbiter (Phase 4 Specification):
            int k_pred = 1;
            int k_actual = 1;
            if (seq_id < (llama_seq_id) arbiters.size()) {
                arbiters[seq_id].feed_token_id(dp.id_last);

                k_pred = arbiters[seq_id].pred_k();
                k_actual = arbiters[seq_id].select_k();

                const char * env_force_k = std::getenv("LLAMA_SPEC_FORCE_K");
                if (env_force_k) {
                    k_actual = std::atoi(env_force_k);
                }

                last_k_drafted[seq_id] = k_actual;
                cycle_count[seq_id]++;

                log_arbiter(seq_id, k_pred, k_actual);
            }

            if (k_actual == 0) {
                last_k_drafted[seq_id] = 0;
                continue;
            }

            common_sampler_reset(smpls[seq_id].get());

            const int32_t n = (int32_t) dp.n_past;

            const int32_t k_eff   = std::min(k_actual, (int32_t) params.n_max);
            int32_t       n_draft = k_eff;
            if (pipe_attached && !pstats.disabled && seq_id == 0 && k_eff > 0 && pipe_lookahead_allowed()) {
                // P1 gate: pay the lookahead tax (one extra block row) only when the expected overlap gain covers it
                if (k_eff + 1 <= (int32_t) params.n_max && arbiters[seq_id].lookahead_worth(k_eff)) {
                    n_draft              = k_eff + 1;
                    la_requested[seq_id] = 1;
                    k_eff_seq[seq_id]    = k_eff;
                    pstats.la_on_seq++;
                } else {
                    pstats.la_off_seq++;
                }
            }

            const int32_t n_block_tokens = n_draft + (is_dspark && sample_from_anchor ? 0 : 1);
            i_block_beg[seq_id] = batch.n_tokens;
            n_block    [seq_id] = n_block_tokens;
            for (int32_t i = 0; i < n_block_tokens; ++i) {
                common_batch_add(batch, i == 0 ? dp.id_last : mask_token_id, n + i, { seq_id }, !is_dflash2);
            }
        }

        if (batch.n_tokens == 0) {
            return;
        }

        // decode all sequence's noise block in a single batch
        if (spec::g_dflash_prof.active) {
            spec::g_dflash_prof.reset((int32_t) batch.n_tokens);
        }
        const int64_t t_draft_start_us = ggml_time_us();
        int ret = llama_decode(ctx_dft, batch);
        const int64_t t_draft_enq_us = ggml_time_us() - t_draft_start_us;

        const int64_t t_sync_start_us = ggml_time_us();
        llama_synchronize(ctx_dft);
        const int64_t t_draft_sync_us = ggml_time_us() - t_sync_start_us;
        const int64_t t_draft_end_us  = ggml_time_us();

        const float enq_ms  = t_draft_enq_us / 1000.0f;
        const float sync_ms = t_draft_sync_us / 1000.0f;
        const float total_d_ms = (t_draft_end_us - t_draft_start_us) / 1000.0f;
        LOG_INF("[dflash-timer] batch_tokens=%d draft_cpu_enq=%.2f ms draft_gpu_sync=%.2f ms total_td=%.2f ms\n",
                (int)batch.n_tokens, enq_ms, sync_ms, total_d_ms);
        if (spec::g_dflash_prof.active && spec::g_dflash_prof.t_output_us > 0) {
            spec::g_dflash_prof.dur_sel_us = t_draft_end_us - spec::g_dflash_prof.t_output_us;
            LOG_INF("[dflash-breakdown] batch_tokens=%d | layers (norm)=%6.2f ms | lm_head (output)=%6.2f ms | selector (lattice)=%6.2f ms | total=%6.2f ms\n",
                    spec::g_dflash_prof.last_n_tokens,
                    spec::g_dflash_prof.dur_layers_us / 1000.0f,
                    spec::g_dflash_prof.dur_head_us   / 1000.0f,
                    spec::g_dflash_prof.dur_sel_us    / 1000.0f,
                    (t_draft_end_us - spec::g_dflash_prof.t_start_us) / 1000.0f);
            if (spec::g_dflash_prof.deep_profile) {
                int64_t tot_attn_us = 0, tot_ffn_us = 0;
                for (int il = 0; il < 5; ++il) {
                    tot_attn_us += spec::g_dflash_prof.dur_attn_us[il];
                    tot_ffn_us  += spec::g_dflash_prof.dur_ffn_us[il];
                    LOG_INF("[dflash-layer-%d] Attn(+QKV)=%5.2f ms | FFN=%5.2f ms | LayerTotal=%5.2f ms\n",
                            il,
                            spec::g_dflash_prof.dur_attn_us[il]  / 1000.0f,
                            spec::g_dflash_prof.dur_ffn_us[il]   / 1000.0f,
                            spec::g_dflash_prof.dur_layer_us[il] / 1000.0f);
                }
                LOG_INF("[dflash-deep-summary] 5-Layer Sum: Attn(+QKV)=%5.2f ms | FFN=%5.2f ms | Conv/Residual/Other=%5.2f ms\n",
                        tot_attn_us / 1000.0f,
                        tot_ffn_us  / 1000.0f,
                        (spec::g_dflash_prof.dur_layers_us - tot_attn_us - tot_ffn_us) / 1000.0f);
            }
        }
        if (ret != 0) {
            LOG_WRN("%s: llama_decode returned %d\n", __func__, ret);
            return;
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_block_beg[seq_id] < 0) {
                continue;
            }
            if (seq_id < (llama_seq_id) last_draft_ms.size()) {
                last_draft_ms[seq_id] = total_d_ms;
            }
            auto & dp = dparams[seq_id];

            const int32_t beg            = i_block_beg[seq_id];
            const int32_t n_block_tokens = n_block[seq_id];

            auto & result = *dp.result;

            read_block_result(seq_id, beg, n_block_tokens, result);

            // strip the lookahead row: it is the anchor guess for the next cycle and is never part of the verified draft
            if (la_requested[seq_id]) {
                if (result.size() == (size_t) k_eff_seq[seq_id] + 1) {
                    la_token = result.back();
                    la_k     = k_eff_seq[seq_id];
                    la_valid = true;
                    result.pop_back();
                }
                if (result.size() < (size_t) params.n_min) {
                    result.clear();
                    la_valid = false;
                }
            }

            if (!is_dflash2 && seq_id < (llama_seq_id) last_k_drafted.size()) {
                last_k_drafted[seq_id] = (int32_t) result.size();
            }
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) override {
        if (!is_other && seq_id >= 0 && seq_id < (llama_seq_id) arbiters.size()) {
            float d_ms = seq_id < (llama_seq_id) last_draft_ms.size() ? last_draft_ms[seq_id] : 0.0f;
            arbiters[seq_id].observe(last_k_drafted[seq_id], n_accepted, d_ms);
            accept_count[seq_id]++;
        }
    }
};

struct common_speculative_impl_draft_mtp : public common_speculative_impl {
    common_params_speculative_draft params; // reuses the draft-model params slot (ctx_tgt/ctx_dft)

    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;

    // backend sampler chain per seq, attached to ctx_dft
    std::vector<llama_sampler *> backend_chains;

    int32_t n_embd = 0;

    // One MTP draft driver, three modes (set once in the ctor):
    //   is_mem_shared (gemma4): shares the target KV, runs all heads in one graph.
    //   chain_heads (step35): n_mtp_layers trained heads, one per draft step.
    //   neither (qwen35 / qwen35moe): a single trained MTP head.
    int32_t n_mtp_layers  = 1;
    bool    is_mem_shared = false;   // gemma4
    bool    chain_heads   = false;   // derived in the ctor: n_mtp_layers > 1 && !is_mem_shared

    // Per-sequence cross-batch carryover: pair (h_p, x_{p+1}) at MTP pos p+1.
    // The last h-row of one process() call needs the first token of the NEXT
    // call to pair with, so it's stashed here until that next call fires.
    std::vector<std::vector<float>> pending_h;   // [n_seq][n_embd]

    std::vector<int32_t> i_batch_beg;
    std::vector<int32_t> i_batch_end;

    // Hidden rows from the most recent target verification batch, grouped by seq.
    // Row 0 corresponds to the sampled token, row N to the Nth accepted draft token.
    std::vector<std::vector<float>> verify_h;
    std::vector<int32_t> verify_h_rows;

    std::vector<int>                i_last;
    std::vector<std::vector<float>> chain_h;

    common_speculative_impl_draft_mtp(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_MTP, n_seq)
        , params(params.draft)
    {
        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        GGML_ASSERT(ctx_tgt && ctx_dft && "MTP requires ctx_tgt and ctx_dft to be set");

        n_embd = llama_model_n_embd_out(llama_get_model(ctx_dft));
        GGML_ASSERT(n_embd == llama_model_n_embd_out(llama_get_model(ctx_tgt)) &&
                "MTP input row width must match the target h_nextn width");
        n_mtp_layers = std::max(1, (int) llama_model_n_layer_nextn(llama_get_model(ctx_dft)));

        SPC_TRC("%s", "adding speculative implementation 'draft-mtp'\n");
        SPC_TRC("- n_max=%d, n_min=%d, p_min=%.2f, n_embd=%d, backend_sampling=%d\n", this->params.n_max, this->params.n_min, this->params.p_min, n_embd, (int) this->params.backend_sampling);
        SPC_TRC("- gpu_layers=%d, cache_k=%s, cache_v=%s, ctx_tgt=%s, ctx_dft=%s, devices=[%s]\n",
                this->params.n_gpu_layers,
                ggml_type_name(this->params.cache_type_k),
                ggml_type_name(this->params.cache_type_v),
                ctx_tgt ? "yes" : "no",
                ctx_dft ? "yes" : "no",
                common_speculative_get_devices_str(this->params.devices).c_str());

        const int32_t n_b = (int32_t) llama_n_batch(ctx_dft);
        batch = llama_batch_init(/*n_tokens=*/ n_b, /*embd=*/ n_embd, /*n_seq_max=*/ 1);
        // llama_batch_init allocates only one of token/embd; MTP needs both.
        // TODO: fix, how to call without malloc
        batch.token = (llama_token *) malloc(sizeof(llama_token) * n_b);

        smpls.resize(n_seq);
        for (auto & s : smpls) {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            sparams.top_k    = 10;
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            s.reset(common_sampler_init(llama_get_model(ctx_dft), sparams));
        }

        // offload draft sampling to the backend
        backend_chains.assign(n_seq, nullptr);
        if (this->params.backend_sampling) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(10));

                if (!llama_set_sampler(ctx_dft, seq_id, chain)) {
                    SPC_WRN("backend offload failed for seq_id=%d; using CPU sampler\n", (int) seq_id);
                    llama_sampler_free(chain);
                    chain = nullptr;
                }
                backend_chains[seq_id] = chain;
            }
        }

        llama_set_embeddings_nextn(ctx_tgt, true, /*masked*/ false);
        llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ true);

        is_mem_shared = llama_get_ctx_other(ctx_dft) == ctx_tgt;
        chain_heads   = n_mtp_layers > 1 && !is_mem_shared;

        if (chain_heads) {
            this->params.n_max = std::min(this->params.n_max, n_mtp_layers);

            chain_h.assign(n_seq, {});
            for (auto & c : chain_h) {
                c.reserve((size_t) (this->params.n_max + 1) * n_embd);
            }
        }

        pending_h.assign(n_seq, std::vector<float>(n_embd, 0.0f));

        i_last.assign(n_seq, -1);
        i_batch_beg.assign(n_seq, -1);
        i_batch_end.assign(n_seq, -1);

        verify_h.assign(n_seq, {});
        verify_h_rows.assign(n_seq, 0);
    }

    ~common_speculative_impl_draft_mtp() override {
        auto * ctx_dft = this->params.ctx_dft;
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) backend_chains.size(); ++seq_id) {
            if (backend_chains[seq_id] == nullptr) {
                continue;
            }
            if (ctx_dft) {
                llama_set_sampler(ctx_dft, seq_id, nullptr);
            }
            llama_sampler_free(backend_chains[seq_id]);
        }
        backend_chains.clear();

        if (batch.token != nullptr) {
            free(batch.token);
            batch.token = nullptr;
        }
        llama_batch_free(batch);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }

        auto * ctx_dft = this->params.ctx_dft;
        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);

        if (pos_max < N - 1 && !is_mem_shared) {
            SPC_WRN("ctx_dft pos_max=%d < N-1=%d - "
                    "process() hook may not have run on every prefill ubatch "
                    "(need_embd / logits=1 on every prompt position?). "
                    "Drafts may degrade.\n",
                    (int) pos_max, N - 1);
        }
    }

    bool process(const llama_batch & batch_in) override {
        if (batch_in.n_tokens <= 0) {
            return true;
        }

        // TODO: how to make it work with vision tokens?
        if (batch_in.token == nullptr || batch_in.embd != nullptr) {
            return true;
        }

        const int32_t n_tokens = batch_in.n_tokens;

        // remember the frist and last batch index for each sequence
        std::fill(i_batch_beg.begin(), i_batch_beg.end(), -1);
        std::fill(i_batch_end.begin(), i_batch_end.end(), -1);

        for (int k = 0; k < n_tokens; ++k) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                GGML_ASSERT(batch_in.n_seq_id[k] == 1);

                if (batch_in.seq_id[k][0] == seq_id) {
                    i_batch_end[seq_id] = k;
                    if (i_batch_beg[seq_id] < 0) {
                        i_batch_beg[seq_id] = k;
                    }
                }
            }
        }

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;

        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        // if kv is shared with target (e.g Gemma4), then we can skip this catch-up decode
        if (!is_mem_shared) {
            common_batch_clear(batch);

            for (int k = 0; k < n_tokens; ++k) {
                common_batch_add(batch, batch_in.token[k], batch_in.pos[k], { batch_in.seq_id[k][0] }, 0);
            }

            // shift the tgt embeddings to the right by one position
            // assumes that the tokens in the batch are sequential for each sequence
            // i.e. we cannot have seq_id like this: [0, 0, 0, 1, 1, 0, 1, 1]
            //                                                       ^--- this is a problem
            // TODO:this is generally true, but would be nice to assert it
            {
                const float * h_tgt = llama_get_embeddings_nextn(ctx_tgt);
                std::memcpy(batch.embd + (size_t) 1 * n_embd, h_tgt, row_bytes * (n_tokens-1));
            }

            // fill the pending embeddings from a previous run
            auto set_h = [&](int idx, const float * h_row) {
                std::memcpy(batch.embd + (size_t) idx * n_embd, h_row, row_bytes);
            };

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (i_batch_beg[seq_id] < 0) {
                    continue;
                }

                set_h(i_batch_beg[seq_id], pending_h[seq_id].data());
            }

            auto * mem_dft = llama_get_memory(ctx_dft);

            bool ok = true;
            for (int head = 0; head < n_mtp_layers; ++head) {
                if (chain_heads) {
                    // ref: https://github.com/ggml-org/llama.cpp/pull/24340/changes#r3413498544
                    for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                        if (i_batch_beg[seq_id] < 0) {
                            continue;
                        }
                        llama_memory_seq_rm(mem_dft, seq_id, batch_in.pos[i_batch_beg[seq_id]], -1);
                    }
                    llama_set_nextn_layer_offset(ctx_dft, head);
                }

                const int32_t rc = llama_decode(ctx_dft, batch);
                if (rc != 0) {
                    SPC_ERR("llama_decode(ctx_dft) head=%d failed rc=%d (pos=%d)\n",
                            head, (int) rc, (int) batch_in.pos[0]);
                    ok = false;
                    break;
                }
            }

            if (chain_heads) {
                llama_set_nextn_layer_offset(ctx_dft, 0); // restore default for non-draft decodes
            }
            if (!ok) {
                return false;
            }
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_batch_end[seq_id] < 0) {
                continue;
            }

            const int32_t n_rows = i_batch_end[seq_id] - i_batch_beg[seq_id] + 1;
            verify_h_rows[seq_id] = n_rows;
            verify_h[seq_id].resize((size_t) n_rows * n_embd);

            for (int32_t i = 0; i < n_rows; ++i) {
                const float * h = llama_get_embeddings_nextn_ith(ctx_tgt, i_batch_beg[seq_id] + i);
                std::memcpy(verify_h[seq_id].data() + (size_t) i * n_embd, h, row_bytes);
            }

            std::memcpy(pending_h[seq_id].data(),
                    verify_h[seq_id].data() + (size_t) (n_rows - 1) * n_embd, row_bytes);
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            common_batch_add(batch, dp.id_last, dp.n_past, { seq_id }, true);
            std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd, pending_h[seq_id].data(), row_bytes);

            i_last[seq_id] = batch.n_tokens - 1;

            if (chain_heads) {
                chain_h[seq_id].assign(pending_h[seq_id].begin(), pending_h[seq_id].end());
            }
        }

        int i = 0;

        while (n_drafting > 0) {
            // each step decodes under a different head, i.e. a different decoder layer, and
            // KV is per layer. process() filled this layer's KV only for positions < n_past
            // (prompt + accepted prefix) — nothing in the draft region yet. so reset the
            // draft region (the seq_rm lower bound is n_past, leaving the prompt KV intact)
            // and select head i so it rebuilds its own layer's KV there; decoding just the
            // latest token would leave its attention reading cells only another head wrote.
            if (chain_heads) {
                auto * mem_dft = llama_get_memory(ctx_dft);
                for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                    if (drafting[seq_id]) {
                        llama_memory_seq_rm(mem_dft, seq_id, dparams[seq_id].n_past, -1);
                    }
                }
                llama_set_nextn_layer_offset(ctx_dft, i);
            }

            int ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                SPC_ERR("llama_decode[%d] returned %d\n", i, ret);
                break;
            }

            // rebuild the batch for the next step: the growing-KV paths re-add only the
            // new token (the KV already holds the prefix), while chained heads re-add the
            // whole prefix at the next head. dropped sequences are simply not re-added.
            common_batch_clear(batch);

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                common_sampler_sample(smpl, ctx_dft, i_last[seq_id], true);
                const float * h_row = llama_get_embeddings_nextn_ith(ctx_dft, i_last[seq_id]);

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    SPC_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                // add drafted token for each sequence
                const llama_token id = cur_p->data[0].id;

                // only collect very high-confidence draft tokens
                if (cur_p->data[0].p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);

                if (params.n_max <= (int) result.size()) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                if (chain_heads) {
                    // ref: https://github.com/ggml-org/llama.cpp/pull/24340#discussion_r3448031546
                    chain_h[seq_id].insert(chain_h[seq_id].end(), h_row, h_row + n_embd);

                    const int n_rows = (int) result.size() + 1; // id_last + tokens drafted so far
                    for (int t = 0; t < n_rows; ++t) {
                        const llama_token tok = (t == 0) ? dp.id_last : result[t - 1];
                        common_batch_add(batch, tok, dp.n_past + t, { seq_id }, t == n_rows - 1);
                        std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd,
                                    chain_h[seq_id].data() + (size_t) t * n_embd, row_bytes);
                    }
                } else if (is_mem_shared) {
                    // note: with shared memory (e.g. Gemma4 assistants) we use the same position for all draft tokens
                    // ref: https://github.com/huggingface/transformers/blob/effde20942e3f82a1b97449f60b3a48c5ff96145/docs/source/en/model_doc/gemma4_assistant.md?plain=1#L36-L37
                    common_batch_add(batch, id, dp.n_past, { seq_id }, true);
                    std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd, h_row, row_bytes);
                } else {
                    common_batch_add(batch, id, dp.n_past + i + 1, { seq_id }, true);
                    std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd, h_row, row_bytes);
                }

                i_last[seq_id] = batch.n_tokens - 1;
            }

            if (batch.n_tokens == 0) {
                break;
            }

            ++i;
        }

        if (chain_heads) {
            llama_set_nextn_layer_offset(ctx_dft, 0); // restore default for non-draft decodes
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool /*is_other*/) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        const int32_t n_rows = verify_h_rows[seq_id];
        if (n_rows <= 0) {
            return;
        }

        const int32_t i_h = std::min<int32_t>(n_accepted, n_rows - 1);
        const size_t row_bytes = (size_t) n_embd * sizeof(float);
        std::memcpy(pending_h[seq_id].data(), verify_h[seq_id].data() + (size_t) i_h * n_embd, row_bytes);
    }
};

// state of self-speculation (simple implementation, not ngram-map)
struct common_speculative_impl_ngram_simple : public common_speculative_impl {
    common_params_speculative_ngram_map params;

    // shared across all sequences
    common_ngram_simple_config config;

    common_speculative_impl_ngram_simple(
            const common_params_speculative & params, uint32_t n_seq,
            common_ngram_simple_config config)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, n_seq)
        , params(params.ngram_simple)
        , config(config)
    {
        SPC_TRC("%s", "adding speculative implementation 'ngram-simple'\n");
        SPC_TRC("- size_n=%d, size_m=%d, min_hits=%d\n",
                this->params.size_n, this->params.size_m, this->params.min_hits);
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            *dp.result = common_ngram_simple_draft(config, *dp.prompt, dp.id_last);
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }
};

struct common_speculative_impl_ngram_map_k : public common_speculative_impl {
    // n_seq configs
    std::vector<common_ngram_map> config;

    common_speculative_impl_ngram_map_k(
            const common_ngram_map & config,
            uint32_t n_seq)
        : common_speculative_impl(config.key_only ? COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K
            : COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, n_seq)
    {
        for (uint32_t i = 0; i < n_seq; i++) {
            this->config.push_back(config);
        }

        SPC_TRC("adding speculative implementation '%s'\n", common_speculative_type_to_str(this->type).c_str());
        SPC_TRC("- size_key=%d, size_value=%d, key_only=%d, min_hits=%d\n",
                config.size_key, config.size_value, config.key_only, config.min_hits);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        GGML_ASSERT(seq_id < (llama_seq_id) n_seq);

        common_ngram_map_begin(config[seq_id], prompt);
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            common_ngram_map_draft(config[seq_id], *dp.prompt, dp.id_last, *dp.result);
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) override {
        GGML_ASSERT((seq_id < (llama_seq_id) config.size()));

        if (is_other) {
            return;
        }

        common_ngram_map_accept(config[seq_id], n_accepted);
    }
};

struct common_speculative_impl_ngram_mod : public common_speculative_impl {
    common_params_speculative_ngram_mod params;

    // shared across all sequences
    common_ngram_mod mod;

    // enable trace logging if LLAMA_TRACE is set
    const bool verbose;

    struct seq_info {
        // the last position in the prompt that was added to the ngram container
        size_t i_last = 0;

        // length of the last drafted n-gram (number of tokens returned by draft)
        size_t n_draft_last = 0;

        // consecutive accept rounds with low acceptance fraction (< 0.5)
        int n_low = 0;
    };

    std::vector<seq_info> sinfos;

    common_speculative_impl_ngram_mod(
            const common_params_speculative & params,
            uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_MOD, n_seq)
        , params(params.ngram_mod)
        , mod(params.ngram_mod.n_match, 4*1024*1024)
        , verbose(std::getenv("LLAMA_TRACE") != nullptr) {
        static_assert(sizeof(llama_token) == sizeof(common_ngram_mod::entry_t));

        SPC_TRC("%s", "adding speculative implementation 'ngram-mod'\n");
        SPC_TRC("- n_match=%d, n_max=%d, n_min=%d\n",
                this->params.n_match, this->params.n_max, this->params.n_min);
        SPC_TRC("- mod size=%zu (%.3f MB)\n",
                mod.size(), (float)(mod.size_bytes())/1024/1024);

        if (this->params.n_match < 16) {
            SPC_WRN("ngram_mod n_match=%d is too small - poor quality is possible, "
                    "see: https://github.com/ggml-org/llama.cpp/pull/19164\n", this->params.n_match);
        }

        sinfos.resize(n_seq);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        auto & sinfo = sinfos[seq_id];

        sinfo.i_last = 0;
        sinfo.n_draft_last = 0;

        const size_t n = mod.get_n();
        if (prompt.size() < n) {
            return;
        }

        for (size_t i = 0; i < prompt.size() - n; ++i) {
            mod.add(prompt.data() + i);
        }

        sinfo.i_last = prompt.size() - n;

        const double f = (double)mod.get_used() / (double)mod.size();
        SPC_TRC("ngram_mod occupancy = %zu/%zu (%.2f)\n", mod.get_used(), mod.size(), f);

        constexpr double f_thold = 0.25;
        if (f > f_thold) {
            SPC_WRN("ngram_mod occupancy %.2f exceeds threshold (%.2f) - resetting\n", f, f_thold);

            mod.reset();
        }
    }

    void draft_one(
            llama_seq_id seq_id,
            common_speculative_draft_params & dparams) {
        auto & sinfo = sinfos[seq_id];
        auto & result = *dparams.result;

        const auto & prompt = *dparams.prompt;

        sinfo.n_draft_last = 0;

        const size_t cur_len = prompt.size();
        if (cur_len < mod.get_n()) {
            return;
        }

        const size_t n = mod.get_n();

        // add new ngrams in chunks
        if (sinfo.i_last + 32 < cur_len) {
            for (size_t i = sinfo.i_last; i < cur_len - n; ++i) {
                mod.add(prompt.data() + i);
            }

            sinfo.i_last = cur_len - n;
        }

        result.resize(n + params.n_max);
        for (size_t i = 0; i < n - 1; ++i) {
            result[i] = prompt.at(cur_len - n + 1 + i);
        }
        result[n - 1] = dparams.id_last;

        for (int i = 0; i < params.n_max; ++i) {
            const llama_token token = mod.get(result.data() + i);
            if (token == common_ngram_mod::EMPTY) {
                if (i < params.n_min) {
                    result.clear();
                    return;
                }

                result.resize(n + i);
                break;
            }
            result[n + i] = token;
        }

        // only return the m tokens that were drafted
        for (size_t i = 0; n + i < result.size(); ++i) {
            result[i] = result[n + i];
        }
        result.resize(result.size() - n);

        // store length of drafted n-gram for later acceptance analysis
        sinfo.n_draft_last = result.size();
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            draft_one(seq_id, dp);
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) override {
        if (is_other) {
            return;
        }

        auto & sinfo = sinfos[seq_id];

        // compute acceptance fraction if we have a recorded draft length
        if (sinfo.n_draft_last > 0) {
            const double f_acc = (double)n_accepted / (double)sinfo.n_draft_last;
            if (f_acc < 0.25) {
                sinfo.n_low++;
                if (sinfo.n_low >= 5) {
                    if (verbose) {
                        SPC_TRC("low acceptance streak (%d) - resetting ngram_mod\n", sinfo.n_low);
                    }

                    mod.reset();
                    sinfo.n_low = 0;
                    sinfo.i_last = 0;
                }
            } else {
                sinfo.n_low = 0;
            }
        }
    }
};

struct common_speculative_impl_ngram_cache : public common_speculative_impl {
    common_params_speculative_ngram_cache params;

    uint16_t n_draft;

    bool save_dynamic;
    bool save_static;

    struct seq_info {
        size_t cache_size = 0; // number of tokens in n-gram cache

        common_ngram_cache ngram_cache_context;
        common_ngram_cache ngram_cache_dynamic;
        common_ngram_cache ngram_cache_static;
    };

    std::vector<seq_info> sinfos;

    common_speculative_impl_ngram_cache(
            const common_params_speculative & params,
            uint32_t n_seq,
            uint16_t n_draft,
            const std::string & path_static,
            const std::string & path_dynamic,
            bool save_dynamic,
            bool save_static)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE, n_seq)
        , params(params.ngram_cache)
        , n_draft(n_draft)
        , save_dynamic(save_dynamic)
        , save_static(save_static)
    {
        SPC_TRC("%s", "adding speculative implementation 'ngram-cache'\n");
        SPC_TRC("- n_draft=%d, cache_static=%s, cache_dynamic=%s\n",
                n_draft,
                path_static.empty() ? "none" : path_static.c_str(),
                path_dynamic.empty() ? "none" : path_dynamic.c_str());

        sinfos.resize(n_seq);

        if (!path_static.empty()) {
            try {
                auto ngram_cache_static = common_ngram_cache_load(path_static);

                for (auto & sinfo : sinfos) {
                    sinfo.ngram_cache_static = ngram_cache_static;
                }
            } catch (...) {
                SPC_ERR("failed to open static lookup cache: %s", path_static.c_str());
                GGML_ABORT("Couldn't read static lookup cache");
            }
        }

        if (!path_dynamic.empty()) {
            try {
                auto ngram_cache_dynamic = common_ngram_cache_load(path_dynamic);

                for (auto & sinfo : sinfos) {
                    sinfo.ngram_cache_dynamic = ngram_cache_dynamic;
                }
            } catch (...) {
                SPC_ERR("failed to open dynamic lookup cache: %s", path_dynamic.c_str());
                GGML_ABORT("Couldn't read dynamic lookup cache");
            }
        }
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    void draft_one(
            llama_seq_id seq_id,
            common_speculative_draft_params & dparams) {
        auto & sinfo = sinfos[seq_id];
        auto & result = *dparams.result;

        const auto & prompt = *dparams.prompt;

        if (sinfo.cache_size < prompt.size() + 1) {
            llama_tokens tokens_new;
            tokens_new.reserve(prompt.size() + 1 - sinfo.cache_size);
            for (size_t j = sinfo.cache_size; j < prompt.size(); ++j) {
                tokens_new.push_back(prompt[j]);
            }
            tokens_new.push_back(dparams.id_last); // add the last token

            // Update context ngram cache with new dparams.prompt:
            common_ngram_cache_update(
                    sinfo.ngram_cache_context,
                    LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                    tokens_new, tokens_new.size(), false);
            sinfo.cache_size = prompt.size() + 1;
        }

        llama_tokens inp;
        inp.reserve(prompt.size() + 1);
        for (size_t j = 0; j < prompt.size(); ++j) {
            inp.push_back(prompt[j]);
        }
        inp.push_back(dparams.id_last);

        result.push_back(dparams.id_last);

        common_ngram_cache_draft(
                inp, result, n_draft, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                sinfo.ngram_cache_context,
                sinfo.ngram_cache_dynamic,
                sinfo.ngram_cache_static);

        if (result.size() > 0) {
            // delete first token in result (which is the id_last token)
            result.erase(result.begin());
        }
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            draft_one(seq_id, dp);
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }
};

struct common_speculative {
    common_speculative_draft_params_vec dparams;

    // list of implementations to use and their states
    std::vector<std::unique_ptr<common_speculative_impl>> impls;

    // which implementaion was used for a given seq_id
    std::vector<common_speculative_impl *> impl_last;
};

static common_ngram_map get_common_ngram_map(
        common_speculative_type type,
        const common_params_speculative_ngram_map & config) {
    uint16_t size_key   = config.size_n;
    uint16_t size_value = config.size_m;
    bool     key_only   = type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K;
    uint16_t min_hits   = config.min_hits;

    return common_ngram_map(size_key, size_value, key_only, min_hits);
}

static common_speculative_impl_ngram_cache create_state_ngram_cache(
        const common_speculative_config & config,
        uint32_t n_seq,
        const std::string & path_static,
        const std::string & path_dynamic) {
    uint16_t n_draft = 8; // TODO get from config?

    // TODO bool param in common/common.h to set save_static/save_dynamic?
    bool save_static = false;
    bool save_dynamic = false;

    common_speculative_impl_ngram_cache state(config.params, n_seq, n_draft, path_static, path_dynamic, save_static, save_dynamic);

    return state;
}

std::string common_speculative_type_name_str(const std::vector<common_speculative_type> & types) {
    std::string result;

    for (size_t i = 0; i < types.size(); i++) {
        if (i > 0) {
            result += ",";
        }
        result += common_speculative_type_to_str(types[i]);
    }
    return result;
}

const char * common_speculative_all_types_str() {
    static std::string all_types_str = []() {
        std::vector<common_speculative_type> types;
        types.reserve(COMMON_SPECULATIVE_TYPE_COUNT);
        for (int i = 0; i < COMMON_SPECULATIVE_TYPE_COUNT; i++) {
            types.push_back((common_speculative_type) i);
        }
        return common_speculative_type_name_str(types);
    }();
    return all_types_str.c_str();
}

std::string common_speculative_type_to_str(common_speculative_type type) {
    switch (type) {
        case COMMON_SPECULATIVE_TYPE_NONE:          return "none";
        case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE:  return "draft-simple";
        case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3:  return "draft-eagle3";
        case COMMON_SPECULATIVE_TYPE_DRAFT_MTP:     return "draft-mtp";
        case COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH:  return "draft-dflash";
        case COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK:  return "draft-dspark";
        case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:  return "ngram-simple";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:   return "ngram-map-k";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: return "ngram-map-k4v";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:     return "ngram-mod";
        case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:   return "ngram-cache";
        default:                                    return "unknown";
    }
}

std::vector<common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names) {
    std::vector<common_speculative_type> types;
    types.reserve(names.size());

    for (const auto & name : names) {
        auto type = common_speculative_type_from_name_map.find(name);
        if (type != common_speculative_type_from_name_map.end()) {
            if (type->second == COMMON_SPECULATIVE_TYPE_NONE) {
                return std::vector<common_speculative_type> { COMMON_SPECULATIVE_TYPE_NONE };
            }
            types.push_back(type->second);
            continue;
        }
        throw std::invalid_argument("unknown speculative type: " + name);
    }

    return types;
}

common_speculative_type common_speculative_type_from_name(const std::string & name) {
    const auto it = common_speculative_type_from_name_map.find(name);
    if (it == common_speculative_type_from_name_map.end()) {
        return COMMON_SPECULATIVE_TYPE_COUNT;
    }
    return it->second;
}

std::vector<common_speculative_type> common_speculative_types_from_gguf(const std::string & path) {
    struct gguf_init_params gguf_params = {
        /* .no_alloc = */ true,
        /* .ctx      = */ nullptr,
    };

    gguf_context_ptr gguf_ctx(gguf_init_from_file(path.c_str(), gguf_params));
    if (!gguf_ctx) {
        return {};
    }

    const int64_t arch_id = gguf_find_key(gguf_ctx.get(), "general.architecture");
    if (arch_id < 0 || gguf_get_kv_type(gguf_ctx.get(), arch_id) != GGUF_TYPE_STRING) {
        return {};
    }

    const std::string arch = gguf_get_val_str(gguf_ctx.get(), arch_id);
    if (arch != "dflash") {
        const uint32_t block_count = gguf_get_val_u32(gguf_ctx.get(), gguf_find_key(gguf_ctx.get(), (arch + ".block_count").c_str()));

        if (gguf_find_tensor(gguf_ctx.get(), ("blk." + std::to_string(block_count - 1) + ".nextn.eh_proj.weight").c_str()) >= 0) {
            return { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
        }

        return {};
    }

    // the Markov head distinguishes draft-dspark from draft-dflash
    const auto type = gguf_find_tensor(gguf_ctx.get(), "markov_w1.weight") >= 0
                    ? COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK
                    : COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH;

    SPC_INF("auto-detected speculative type '%s' from the draft model metadata\n", common_speculative_type_to_str(type).c_str());

    return { type };
}

static uint32_t common_get_enabled_speculative_configs(const std::vector<common_speculative_type> & configs) {
    uint32_t result = 0;
    for (size_t i = 0; i < configs.size(); i++) {
        result |= (1u << configs[i]);
    }
    return result;
}

int32_t common_speculative_n_max(const common_params_speculative * spec) {
    int32_t n_max = 0;

    for (const auto type : spec->types) {
        switch (type) {
            case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE:
            case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3:
            case COMMON_SPECULATIVE_TYPE_DRAFT_MTP:
            case COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH:
            case COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK:
                n_max = std::max(n_max, std::max(0, spec->draft.n_max));
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:
                n_max = std::max(n_max, (int32_t) spec->ngram_simple.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:
                n_max = std::max(n_max, (int32_t) spec->ngram_map_k.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V:
                n_max = std::max(n_max, (int32_t) spec->ngram_map_k4v.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:
                n_max = std::max(n_max, std::max(0, spec->ngram_mod.n_max));
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:
                n_max = std::max(n_max, (int32_t) 8);
                break;
            case COMMON_SPECULATIVE_TYPE_NONE:
            case COMMON_SPECULATIVE_TYPE_COUNT:
                break;
        }
    }

    return n_max;
}

common_params common_base_params_to_speculative(const common_params & params) {
    const bool has_draft = params.speculative.has_dft();

    const auto & params_spec = params.speculative.draft;
    common_params result = params;

    result.embedding    = false;
    result.pooling_type = LLAMA_POOLING_TYPE_UNSPECIFIED;

    if (has_draft) {
        result.devices               = params_spec.devices;
        result.model                 = params_spec.mparams;
        result.n_gpu_layers          = params_spec.n_gpu_layers;
        result.tensor_buft_overrides = params_spec.tensor_buft_overrides;

        if (params_spec.cpuparams.n_threads > 0) {
            result.cpuparams.n_threads       = params_spec.cpuparams.n_threads;
            result.cpuparams_batch.n_threads = params_spec.cpuparams_batch.n_threads;
        }
    }

    result.cache_type_k  = params_spec.cache_type_k;
    result.cache_type_v  = params_spec.cache_type_v;

    // the K-cache mean-center bias is calibrated for the target model
    // (per-head/channel K layout); the draft model has a different K
    // geometry, so it must not inherit the bias
    result.kv_mean_center_path.clear();
    result.n_outputs_max = params.n_parallel;
    result.n_outputs_max_per_seq = 1;

    // dflash/dspark decode the whole noise block in a single pass and sample every block position on the backend
    // TODO: refactor such properties to be announced by the speculative types
    //       something like `struct common_speculative_type_props common_speculative_type_get_props(...);`
    const bool has_block_draft = std::any_of(
        params.speculative.types.begin(), params.speculative.types.end(),
        [](common_speculative_type t) {
            return t == COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH || t == COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK;
        });
    if (has_block_draft) {
        // per-seq output positions: DFlash decodes anchor + n_max masks (n_max + 1); DSpark n_max -> +1 covers both
        const int32_t per_seq = std::max(1, params_spec.n_max + 1);
        result.n_outputs_max = params.n_parallel * per_seq;
        if (params_spec.backend_sampling) {
            result.n_outputs_max_per_seq = per_seq;
        }
    }

    return result;
}

struct common_speculative_init_result::impl {
    impl() = default;
    ~impl() = default;

    // note: the order in which model, context, etc. are declared matters because their destructors will be called bottom-to-top
    llama_model_ptr   model;
    llama_context_ptr context;
};

common_speculative_init_result::common_speculative_init_result(
    common_params & params,
      llama_model * model_tgt,
    llama_context * ctx_tgt) :
    pimpl(new impl{}) {
    const bool has_draft = params.speculative.has_dft();
    const bool spec_mtp = std::find(params.speculative.types.begin(),
                                    params.speculative.types.end(),
                                    COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params.speculative.types.end();

    auto mparams = common_model_params_to_llama(params);
    auto cparams = common_context_params_to_llama(params);

    if (spec_mtp) {
        cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    }

    // the draft context holds as many tokens per sequence as the target context
    cparams.n_ctx = llama_n_ctx(ctx_tgt);

    // n_rs_seq stays as common_context_params_to_llama set it: the draft context needs the same rollback window as the target, with n_rs_seq == 0 its seq_rm fails silently on partial acceptance and keeps stale positions
    cparams.ctx_other = ctx_tgt;

    std::string model_path;
    if (has_draft) {
        model_path = params.speculative.draft.mparams.path;
        LOG_INF("%s: loading draft model '%s'\n", __func__, model_path.c_str());

        const char * shared_head = std::getenv("LLAMA_DSPARK_SHARED_HEAD");
        if (shared_head && std::strcmp(shared_head, "1") == 0) {
            mparams.dspark_head_source = model_tgt;
        }
        llama_model * model_dft = llama_model_load_from_file(params.model.path.c_str(), mparams);
        if (model_dft == NULL) {
            LOG_ERR("%s: failed to load draft model, '%s'\n", __func__, model_path.c_str());
            return;
        }

        pimpl->model.reset(model_dft);

        const char * greedy_env = std::getenv("LLAMA_DSPARK_GREEDY_IDS");
        if (greedy_env && std::strcmp(greedy_env, "1") == 0) {
            llama_dspark_meta meta;
            if (llama_model_dspark_get_meta(model_dft, &meta) && meta.graph_corrected) {
                // The forward block can exceed the retained verification depth.
                cparams.n_outputs_max_per_seq = meta.block_size;
                cparams.n_outputs_max         = cparams.n_seq_max * meta.block_size;
            }
        }

        const char * env_prof = std::getenv("LLAMA_DFLASH_PROFILE");
        if (env_prof && std::atoi(env_prof) != 0) {
            spec::g_dflash_prof.active = true;
            const char * env_deep = std::getenv("LLAMA_DFLASH_DEEP_PROFILE");
            if (env_deep && std::atoi(env_deep) != 0) {
                spec::g_dflash_prof.deep_profile = true;
                LOG_INF("%s: DFlash DEEP node breakdown profiler enabled (sub-layer profiling)\n", __func__);
            } else {
                LOG_INF("%s: DFlash node breakdown profiler enabled\n", __func__);
            }
            cparams.cb_eval = spec::dflash_breakdown_profiler::cb;
            cparams.cb_eval_user_data = &spec::g_dflash_prof;
        }

        llama_context * ctx_dft = llama_init_from_model(model_dft, cparams);
        if (ctx_dft == nullptr) {
            LOG_ERR("%s: failed to create MTP context\n", __func__);
            return;
        }

        pimpl->context.reset(ctx_dft);
    } else if (spec_mtp) {
        model_path = params.model.path;

        LOG_INF("%s: creating MTP draft context against the target model '%s'\n", __func__, model_path.c_str());

        llama_context * ctx_dft = llama_init_from_model(model_tgt, cparams);
        if (ctx_dft == nullptr) {
            LOG_ERR("%s: failed to create MTP context\n", __func__);
            return;
        }

        pimpl->context.reset(ctx_dft);
    }
}

common_speculative_init_result::~common_speculative_init_result() = default;

llama_model * common_speculative_init_result::model() {
    return pimpl->model.get();
}

llama_context * common_speculative_init_result::context() {
    return pimpl->context.get();
}

common_speculative_init_result_ptr common_speculative_init_from_params(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt) {
    return std::make_unique<common_speculative_init_result>(params, model_tgt, ctx_tgt);
}

common_speculative_output_limits common_speculative_get_output_limits(
        int32_t n_batch, int32_t n_parallel, int32_t n_draft) {
    const int64_t per_seq = 1 + (int64_t) std::max(0, n_draft);
    const int64_t total   = (int64_t) n_parallel * per_seq;

    return {
        /* .total   = */ (int32_t) std::min<int64_t>(n_batch, total),
        /* .per_seq = */ (int32_t) std::min<int64_t>(n_batch, per_seq),
    };
}

// initialization of the speculative decoding system
//
common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq) {
    // Compute the implementations to use based on the config and their order of preference
    std::vector<common_speculative_config> configs = {}; // list of speculative configs to try
    {
        uint32_t enabled_configs = common_get_enabled_speculative_configs(params.types);

        auto add_config_if_enabled = [&](common_speculative_type type, bool available = true) {
            if (available && (enabled_configs & (1u << type))) {
                configs.emplace_back(type, params);
            }
        };

        // when adding a new type - update here the logic above
        static_assert(COMMON_SPECULATIVE_TYPE_COUNT == 11);

        // this list here defines the priority of the speculators
        // the one with highest priority are listed first
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_MOD);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE);

        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3, params.draft.ctx_dft != nullptr);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_MTP,    params.draft.ctx_dft != nullptr);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH, params.draft.ctx_dft != nullptr);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK, params.draft.ctx_dft != nullptr);
    }

    std::vector<std::unique_ptr<common_speculative_impl>> impls = {};

    for (const common_speculative_config & config : configs) {
        switch (config.type) {
            case COMMON_SPECULATIVE_TYPE_NONE:
                break;
            case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_simple>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_eagle3>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_MTP: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_mtp>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_dflash>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK: {
                    char arch[32] = {};
                    llama_model_meta_val_str(llama_get_model(config.params.draft.ctx_dft), "general.architecture", arch,
                                             sizeof(arch));
                    if (std::string(arch) == "dspark") {
                        impls.emplace_back(new common_speculative_impl_draft_dspark(config.params, n_seq));
                        break;
                    }
                impls.push_back(std::make_unique<common_speculative_impl_draft_dflash>(
                        config.params, n_seq, COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE: {
                common_ngram_map ngram_map = get_common_ngram_map(config.type, config.params.ngram_simple);

                uint16_t ngram_size_key   = ngram_map.size_key;
                uint16_t mgram_size_value = ngram_map.size_value;

                auto config_simple = common_ngram_simple_config {
                    /* .size_ngram = */ ngram_size_key,
                    /* .size_mgram = */ mgram_size_value
                };
                auto state = std::make_unique<common_speculative_impl_ngram_simple>(
                    /* .params = */ config.params,
                    /* .n_seq  = */ n_seq,
                    /* .state  = */ config_simple
                );
                impls.push_back(std::move(state));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_map_k>(
                            get_common_ngram_map(config.type, config.params.ngram_map_k), n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_map_k>(
                            get_common_ngram_map(config.type, config.params.ngram_map_k4v), n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_mod>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE: {
                auto state = create_state_ngram_cache(
                        config, n_seq,
                        params.ngram_cache.lookup_cache_static,
                        params.ngram_cache.lookup_cache_dynamic);
                impls.push_back(std::make_unique<common_speculative_impl_ngram_cache>(state));
                break;
            }
            default:
                break;
        }
    }

    if (impls.empty()) {
        SPC_TRC("%s", "no implementations specified for speculative decoding\n");
        return nullptr;
    }

    auto * result = new common_speculative {
        /* .dparams   = */ common_speculative_draft_params_vec(n_seq),
        /* .impls     = */ std::move(impls),
        /* .impl_last = */ std::vector<common_speculative_impl *>(n_seq, nullptr)
    };

    return result;
}

void common_speculative_free(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    delete spec;
}

common_speculative_draft_params & common_speculative_get_draft_params(
        common_speculative * spec,
        llama_seq_id seq_id) {
    GGML_ASSERT(spec);
    GGML_ASSERT(seq_id < (llama_seq_id) spec->dparams.size());

    return spec->dparams[seq_id];
}

void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        common_time_meas tm(impl->t_begin_us, !impl->gen_perf);
        impl->begin(seq_id, prompt);
        impl->n_call_begin++;
    }
}

bool common_speculative_process(common_speculative * spec, const llama_batch & batch) {
    bool result = true;

    if (spec == nullptr) {
        return result;
    }

    for (auto & impl : spec->impls) {
        result = result && impl->process(batch);
    }

    return result;
}

void common_speculative_draft(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    auto & dparams = spec->dparams;

    {
        int n_drafting = 0;

        for (auto & dp : dparams) {
            GGML_ASSERT(!dp.drafting || dp.result->empty());

            if (dp.drafting) {
                n_drafting++;
            }
        }

        if (n_drafting == 0) {
            return;
        }
    }

    for (auto & impl : spec->impls) {
        {
            common_time_meas tm(impl->t_draft_us, !impl->gen_perf);
            impl->draft(dparams);
            impl->n_call_draft++;
        }

        int n_drafting = 0;

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) dparams.size(); ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            auto & result = *dp.result;

            // a new draft has been sampled
            if (dp.drafting && !result.empty()) {
                dp.drafting = false;

                if (dp.n_max > 0) {
                    if (!result.empty() && (int) result.size() > dp.n_max) {
                        SPC_DBG("truncating draft to %d tokens\n", dp.n_max);
                        result.resize(dp.n_max);
                    }
                }

                if (!result.empty()) {
                    SPC_DBG("called impl %s, hist size = %zu, call_count = %zu, gen = %zu\n",
                            common_speculative_type_to_str(impl.get()->type).c_str(), dp.prompt->size(),
                            impl.get()->n_call_draft, result.size());

                    // remember which implementation was used
                    spec->impl_last[seq_id] = impl.get();

                    impl->n_gen_drafts++;
                    impl->n_gen_tokens += result.size();
                }
            }

            if (dp.drafting) {
                n_drafting++;
            }
        }

        if (n_drafting == 0) {
            break;
        }
    }

    // these sequences failed to generate a draft
    for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) dparams.size(); ++seq_id) {
        auto & dp = dparams[seq_id];

        if (dp.drafting) {
            dp.drafting = false;
        }
    }
}

void common_speculative_accept(common_speculative * spec, llama_seq_id seq_id, uint16_t n_accepted) {
    common_speculative_impl * impl = spec->impl_last[seq_id];

    if (impl == nullptr) {
        GGML_ASSERT(n_accepted == 0);
        return;
    }

    {
        common_time_meas tm(impl->t_accept_us, !impl->gen_perf);

        if (impl->n_acc_tokens_per_pos.size() < n_accepted) {
            impl->n_acc_tokens_per_pos.resize(n_accepted, 0);
        }

        for (size_t i = 0; i < n_accepted; ++i) {
            impl->n_acc_tokens_per_pos[i]++;
        }

        if (n_accepted > 0) {
            impl->n_acc_drafts++;
            impl->n_acc_tokens += n_accepted;
        }

        impl->accept(seq_id, n_accepted, false);
        impl->n_call_accept++;
    }

    // accept with the rest of the implementations, using is_other == true
    for (auto & impl_other : spec->impls) {
        if (impl_other.get() != impl) {
            impl_other->accept(seq_id, n_accepted, true);
        }
    }
}

// TODO: support the case of more than one speculative implementations having a state
bool common_speculative_get_state(common_speculative * spec, llama_seq_id seq_id, std::vector<uint8_t> & data) {
    if (spec == nullptr) {
        return false;
    }

    for (auto & impl : spec->impls) {
        if (impl->get_state(seq_id, data)) {
            return true;
        }
    }

    return false;
}

void common_speculative_set_state(common_speculative * spec, llama_seq_id seq_id, const std::vector<uint8_t> & data) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        impl->set_state(seq_id, data);
    }
}

std::vector<enum common_speculative_type> common_speculative_get_types(const common_speculative * spec) {
    std::vector<enum common_speculative_type> types;
    if (spec != nullptr) {
        for (const auto & impl : spec->impls) {
            types.push_back(impl->type);
        }
    }
    return types;
}

void common_speculative_print_stats(const common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    for (const auto & impl : spec->impls) {
        std::string str_perf;
        if (impl->gen_perf) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << impl->t_begin_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_draft_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_accept_us / 1000.0;
            str_perf = ", dur(b,g,a) = " + oss.str() + " ms";
        } else {
            str_perf = "";
        }

        std::string str_stats;
        if (impl->n_call_accept > 0) {
            const double mean =
                1.0 + (double) impl->n_acc_tokens / (double) impl->n_call_accept;
            std::ostringstream tmp;
            tmp << std::fixed << std::setprecision(3);
            for (size_t i = 0; i < impl->n_acc_tokens_per_pos.size(); ++i) {
                if (i > 0) {
                    tmp << ", ";
                }
                tmp << (double) impl->n_acc_tokens_per_pos[i] / (double) impl->n_call_accept;
            }
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << mean;
            str_stats = ", #mean acc len = " + oss.str() + ", #acc rate/pos = (" + tmp.str() + ")";
        }

        SPC_TRC("statistics %16s: #calls(b,g,a) = %4zu %6zu %6zu, #gen drafts = %6zu, #acc drafts = %5zu, #gen tokens = %6zu, #acc tokens = %5zu%s%s\n",
                common_speculative_type_to_str(impl->type).c_str(),
                impl->n_call_begin, impl->n_call_draft, impl->n_call_accept,
                impl->n_gen_drafts,
                impl->n_acc_drafts,
                impl->n_gen_tokens,
                impl->n_acc_tokens,
                str_stats.c_str(),
                str_perf.c_str());
    }
}

bool common_speculative_need_embd_capture(common_speculative * spec) {
    if (spec == nullptr) {
        return false;
    }

    for (auto & impl : spec->impls) {
        if (impl->need_embd_capture()) {
            return true;
        }
    }

    return false;
}

bool common_speculative_dspark_stage_ctx_test(common_speculative * spec,
                                              llama_seq_id         seq_id,
                                              const float *        feat,
                                              int64_t              n_rows,
                                              int64_t              n_embd_cap,
                                              const int32_t *      pos) {
    if (!spec) {
        return false;
    }
    bool staged = false;
    for (auto & impl : spec->impls) {
        staged |= impl->stage_test_ctx_feat(seq_id, feat, n_rows, n_embd_cap, pos);
    }
    return staged;
}
// ---------------------------------------------------------------------------------------------------------------
// Stage S8: DFlash worker pipeline entry points (declared in speculative-tap.h)
// ---------------------------------------------------------------------------------------------------------------
static common_speculative_impl_draft_dflash * common_speculative_find_dflash(common_speculative * spec) {
    if (spec == nullptr) {
        return nullptr;
    }
    for (auto & impl : spec->impls) {
        if (auto * d = dynamic_cast<common_speculative_impl_draft_dflash *>(impl.get())) {
            return d;
        }
    }
    return nullptr;
}

static const common_speculative_impl_draft_dflash * common_speculative_find_dflash(const common_speculative * spec) {
    return common_speculative_find_dflash(const_cast<common_speculative *>(spec));
}

bool common_speculative_pipeline_attach(common_speculative * spec, spec::layer_tap * tap) {
    auto * d = common_speculative_find_dflash(spec);
    return d != nullptr && d->pipe_attach(tap);
}

bool common_speculative_pipeline_arm(common_speculative * spec, llama_seq_id seq_id, llama_pos pos0, int32_t n_real, int32_t k_exec) {
    auto * d = common_speculative_find_dflash(spec);
    return d != nullptr && d->pipe_arm(seq_id, pos0, n_real, k_exec);
}

void common_speculative_pipeline_disarm(common_speculative * spec, int64_t t_decode_end_us) {
    if (auto * d = common_speculative_find_dflash(spec)) {
        d->pipe_disarm(t_decode_end_us);
    }
}

void common_speculative_pipeline_collect(common_speculative * spec, llama_seq_id seq_id, int n_accepted, llama_token id_last_actual) {
    if (auto * d = common_speculative_find_dflash(spec)) {
        d->pipe_collect(seq_id, n_accepted, id_last_actual);
    }
}

bool common_speculative_pipeline_take(common_speculative * spec, llama_seq_id seq_id, int n_draft_max, llama_tokens & draft) {
    auto * d = common_speculative_find_dflash(spec);
    if (d == nullptr) {
        return false;
    }
    {
        common_time_meas tm(d->t_draft_us, !d->gen_perf);
        if (!d->pipe_take(seq_id, n_draft_max, draft)) {
            return false;
        }
    }
    // same per-draft accounting common_speculative_draft() does for a sequential draft
    d->n_call_draft++;
    d->n_gen_drafts++;
    d->n_gen_tokens += draft.size();
    spec->impl_last[seq_id] = d;
    return true;
}

void common_speculative_pipeline_abort(common_speculative * spec) {
    if (auto * d = common_speculative_find_dflash(spec)) {
        d->pipe_abort();
    }
}

::spec::pipe_stats common_speculative_pipeline_get_stats(const common_speculative * spec) {
    const auto * d = common_speculative_find_dflash(spec);
    if (d == nullptr) {
        return ::spec::pipe_stats{};
    }
    ::spec::pipe_stats s = d->pstats;
    if (d->tap != nullptr) {
        s.t_tap_cb_us = d->tap->stats().t_cb_us;
    }
    return s;
}
