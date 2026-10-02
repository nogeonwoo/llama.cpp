#pragma once
// Stage S8 / P0: mid-graph feature tap for DFlash overlap (header-only, C++17).
//
// Mechanism (real API only): llama_context_params.cb_eval -> ggml_backend_sched_set_eval_callback.
// When a callback is installed, ggml_backend_sched_compute_splits() cuts each split into chunks that end
// at every node for which callback(t, ask=true) returned true. The chunk is submitted with
// ggml_backend_graph_compute_async(), the backend is synchronized, and callback(t, ask=false) is invoked on the
// thread that called llama_decode(). Returning true resumes the remaining nodes (the next chunk).
//
// We ask for exactly one node: the one whose value is the input of the highest DFlash extract layer
// (llama_get_embeddings_layer_inp(ctx_tgt, il) == output of layer il-1 == node "l_out-<il-1>", il == 0 ->
// "model.embed_tokens"). All lower extract layers are graph outputs (ggml_set_output in llm_graph_result::
// set_outputs), so their buffers are still intact at that point and are read in the same callback.
// The remaining layers keep running on the target's own Metal queue while a worker thread drives ctx_dft.
//
// Safety net: the tap never trusts itself. The consumer compares every captured row bit-exactly against the
// features llama_decode() later publishes through llama_get_embeddings_layer_inp(); any mismatch disables the
// pipeline and the sequential path is used.

#include "ggml.h"
#include "ggml-backend.h"
#include "speculative.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>

namespace spec {

struct tap_stats {
    uint64_t armed       = 0;  // decodes armed
    uint64_t fired       = 0;  // boundary node reached and rows captured
    uint64_t failed      = 0;  // armed but not captured (name/shape mismatch)
    int64_t  t_cb_us     = 0;  // sum of callback wall time (capture + handoff)
};

class layer_tap {
public:
    // ggml_backend_sched_eval_callback-compatible entry point. user_data must be a layer_tap *.
    static bool eval_cb(struct ggml_tensor * t, bool ask, void * user_data) {
        return static_cast<layer_tap *>(user_data)->on_eval_(t, ask);
    }

    // Name of the graph node that holds llama_get_embeddings_layer_inp(ctx, il).
    static std::string node_name_for_layer_input(uint32_t il) {
        return il == 0 ? std::string("model.embed_tokens") : "l_out-" + std::to_string(il - 1);
    }

    // Called once the DFlash drafter is known (extract layers + target hidden size).
    void configure(const int32_t * layer_ids, uint32_t n_ids, int64_t n_embd) {
        slots_.clear();
        n_embd_    = n_embd;
        last_slot_ = -1;
        uint32_t best = 0;
        for (uint32_t k = 0; k < n_ids; ++k) {
            slot s;
            s.il   = (uint32_t) layer_ids[k];
            s.name = node_name_for_layer_input(s.il);
            if (last_slot_ < 0 || s.il >= best) {
                best       = s.il;
                last_slot_ = (int) k;
            }
            slots_.push_back(std::move(s));
        }
        configured_ = n_ids > 0 && n_embd > 0;
    }

    bool configured() const { return configured_; }

    // Pre-size the host row buffers so that arm() never allocates for batches of up to max_rows rows.
    void reserve_rows(int32_t max_rows) {
        if (!configured_ || max_rows <= 0) {
            return;
        }
        for (auto & s : slots_) {
            s.host.reserve((size_t) max_rows * (size_t) n_embd_);
        }
    }

    size_t   n_slots()        const { return slots_.size(); }
    uint32_t slot_layer(size_t k) const { return slots_[k].il; }
    int64_t  n_embd()         const { return n_embd_; }

    // Host copy of rows [0, n_rows) of extract layer slot k: [n_rows, n_embd] float32.
    const float * rows(size_t k) const { return slots_[k].host.data(); }
    int32_t       n_rows()       const { return n_rows_; }
    bool          captured()     const { return captured_; }

    // Arm before llama_decode(ctx_tgt, batch) on the thread that will call it.
    // n_rows = number of REAL rows (masked-4 dummy rows excluded). on_ready runs inside the callback, after the
    // rows were copied, and must only hand work to another thread (it executes while the GPU is idle).
    void arm(int32_t n_rows, std::function<void()> on_ready) {
        if (!configured_ || n_rows <= 0) {
            return;
        }
        for (auto & s : slots_) {
            s.t = nullptr;
            s.host.resize((size_t) n_rows * (size_t) n_embd_);
        }
        n_rows_   = n_rows;
        captured_ = false;
        on_ready_ = std::move(on_ready);
        owner_    = std::this_thread::get_id();
        armed_    = true;
        ++stats_.armed;
    }

    // Disarm after llama_decode() returns (also clears a tap that never fired).
    void disarm() {
        if (armed_ && !captured_) {
            ++stats_.failed;
        }
        armed_    = false;
        on_ready_ = nullptr;
    }

    const tap_stats & stats() const { return stats_; }

private:
    struct slot {
        uint32_t            il = 0;
        std::string         name;
        ggml_tensor *       t  = nullptr;
        std::vector<float>  host;
    };

    bool on_eval_(struct ggml_tensor * t, bool ask) {
        if (!armed_ || std::this_thread::get_id() != owner_) {
            // never intercept other contexts (drafter) or unarmed decodes
            return ask ? false : true;
        }

        if (ask) {
            // cheap reject: wanted names start with "l_out-" or "model."
            if (t->name[0] != 'l' && t->name[0] != 'm') {
                return false;
            }
            bool is_boundary = false;
            for (size_t k = 0; k < slots_.size(); ++k) {
                auto & s = slots_[k];
                if (s.t == nullptr && std::strcmp(t->name, s.name.c_str()) == 0) {
                    s.t = t;
                    if ((int) k == last_slot_) {
                        is_boundary = true;
                    }
                }
            }
            return is_boundary;
        }

        // ask == false: all nodes up to and including the boundary node finished and the backend is synchronized
        const int64_t t0 = ggml_time_us();
        captured_ = capture_();
        if (captured_) {
            ++stats_.fired;
            if (on_ready_) {
                on_ready_();
            }
        }
        stats_.t_cb_us += ggml_time_us() - t0;
        return true;
    }

    bool capture_() {
        for (auto & s : slots_) {
            const ggml_tensor * t = s.t;
            if (t == nullptr || t->type != GGML_TYPE_F32 || t->ne[0] != n_embd_ || t->ne[1] < n_rows_ ||
                t->nb[0] != sizeof(float) || t->nb[1] != (size_t) n_embd_ * sizeof(float)) {
                return false;
            }
            ggml_backend_tensor_get(t, s.host.data(), 0, (size_t) n_rows_ * (size_t) n_embd_ * sizeof(float));
        }
        return true;
    }

    std::vector<slot>  slots_;
    int                last_slot_  = -1;
    int64_t            n_embd_     = 0;
    int32_t            n_rows_     = 0;
    bool               configured_ = false;
    std::atomic<bool>  armed_{false};
    bool               captured_   = false;
    std::thread::id    owner_;
    std::function<void()> on_ready_;
    tap_stats          stats_;
};

} // namespace spec

// ---------------------------------------------------------------------------------------------------------------
// Pipeline entry points implemented in speculative.cpp (DFlash implementation). All take the opaque spec handle.
// ---------------------------------------------------------------------------------------------------------------
namespace spec {

struct pipe_stats {
    uint64_t submitted      = 0;  // hypotheses handed to the worker
    uint64_t hits           = 0;  // worker draft consumed
    uint64_t miss_accept    = 0;  // not all K drafts accepted
    uint64_t miss_anchor    = 0;  // all accepted, bonus token != lookahead anchor
    uint64_t miss_k         = 0;  // hypothesis K != K the arbiter selected after the update
    uint64_t miss_empty     = 0;  // hypothesis right, worker produced no draft
    uint64_t verify_fail    = 0;  // captured features differ from llama_get_embeddings_layer_inp()
    uint64_t tap_fail       = 0;  // tap armed but no capture
    uint64_t la_on_seq      = 0;  // sequential drafts that paid the lookahead tax
    uint64_t la_off_seq     = 0;  // sequential drafts without lookahead (P1 gate said no)
    uint64_t ingest_jobs    = 0;  // Path B0: ingest-only jobs completed
    int64_t  t_worker_us    = 0;  // sum worker wall time (ingest + draft), hidden when overlap works
    int64_t  t_wait_us      = 0;  // sum time the main thread blocked on the worker after the target finished
    int64_t  t_window_us    = 0;  // sum (target decode end - tap fire): the time available for hiding
    int64_t  t_tap_cb_us    = 0;  // sum tap callback time
    bool     disabled       = false;
};

} // namespace spec

// true when the DFlash implementation was found, LLAMA_SPEC_PIPELINE is on and n_seq == 1; configures `tap`
bool common_speculative_pipeline_attach(common_speculative * spec, spec::layer_tap * tap);

// arm the tap + declare the hypothesis for cycle N+1 right before llama_decode(ctx_tgt, batch) of cycle N.
//   pos0   : position of id_last in the batch (first row)
//   n_real : number of real rows (draft.size() + 1, i.e. WITHOUT the masked-4 dummy row)
//   k_exec : number of drafts verified in this batch
// returns true if a hypothesis will be submitted from the tap callback
bool common_speculative_pipeline_arm(common_speculative * spec, llama_seq_id seq_id, llama_pos pos0, int32_t n_real, int32_t k_exec);

// call after llama_decode()/llama_synchronize() returned (also on failure); `t_decode_end_us` = ggml_time_us()
void common_speculative_pipeline_disarm(common_speculative * spec, int64_t t_decode_end_us);

// call after sampling, BEFORE any other use of ctx_dft: joins the worker, verifies the captured features and decides
// hit/miss. n_accepted = ids.size() - 1, id_last_actual = ids.back().
void common_speculative_pipeline_collect(common_speculative * spec, llama_seq_id seq_id, int n_accepted, llama_token id_last_actual);

// call after common_speculative_accept() at the start of the next cycle: moves the worker draft into `draft` when
// the previous collect() produced a hit and the arbiter still wants the same K. Performs the draft() bookkeeping.
bool common_speculative_pipeline_take(common_speculative * spec, llama_seq_id seq_id, int n_draft_max, llama_tokens & draft);

// drop any in-flight / ready hypothesis (error paths, checkpoint restores)
void common_speculative_pipeline_abort(common_speculative * spec);

spec::pipe_stats common_speculative_pipeline_get_stats(const common_speculative * spec);
