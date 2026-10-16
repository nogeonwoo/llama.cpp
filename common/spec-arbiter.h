// spec-arbiter.h  --  Stage S7 Adaptive Speculative Arbiter for llama.cpp-prism (common/)
//
// Target    : Apple Silicon M3 / Ternary-Bonsai-2-27B (PQ2_0) + DFlash2-ft5-Q4_K_M drafter
// Objective : Direct Expected Throughput Maximization:
//               T_K = (1 + S_K) / C_K,   where S_K = sum_{j=1}^K a_j
//
// Solved Mathematical & Hardware Invariants:
//   1. Physical Hardware Timing (N=5 calibrated on M3):
//        C_0 = 100.5 ms (AR baseline decode)
//        C_1 = 160.5 ms (K=1, M=2 verify)
//        C_2 = 271.7 ms (K=2, M=4 Masked-4 padded verify)
//        C_3 = 278.8 ms (K=3, M=4 dense verify)
//   2. Pairwise Ordering Anomaly Elimination:
//        Direct throughput evaluation T_K prevents K=2 misordering against K=1.
//   3. Statistical Variance Stabilizer:
//        EMA alpha = 0.10 (sigma ~ 0.08) prevents false K=0 park on reasoning (a1 ~ 0.815).
//   4. Periodic Probe Schedule with Exponential Backoff:
//        iv in {16, 32, 64, 128} cycles when parked in K=0.
//
// Header-only, C++17, slot-isolated (one instance per sequence ID).

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

namespace spec {

inline constexpr int arbiter_k_max = 3;

enum class drafter_kind : uint8_t { AUTOREGRESSIVE = 0, BLOCK = 1 };
enum class phase_id     : uint8_t { CONTENT = 0, THINK = 1 };
enum class arbiter_mode : uint8_t { SHADOW = 0, ACTIVE = 1, AUTO = 2 };
enum class mode         : int     { content = 0, think = 1 };

struct arbiter_params {
    double       alpha             = 0.10;  // EMA smoothing factor (stabilized against variance)
    int          warmup            = 3;     // observed cycles before any active K decision
    double       margin            = 0.04;  // promotion hysteresis margin m = 4%
    
    // Hardware-calibrated empirical cycle times (ms) on Apple Silicon M3 (Fused asym-Q2_K workload)
    // Invariant: C_K = tv_K + td_K + th_K (102.7, 153.0, 258.4, 253.4 ms)
    double       C[arbiter_k_max + 1] = { 102.7, 156.8, 274.1, 277.7 };

    uint32_t     probe_iv_min      = 16;    // probe interval when parked in K=0
    uint32_t     probe_iv_max      = 128;
    int          baseline_k        = 1;     // K executed in SHADOW, and in AUTO before warmup
    int          probe_k           = 1;     // K used for probe bursts (1 or 2)
    int          consec_bail       = 2;     // consecutive cycles below baseline before K=0 bailout

    arbiter_mode op_mode           = arbiter_mode::SHADOW;
    drafter_kind drafter           = drafter_kind::BLOCK;
    int          id_open           = -1;
    int          id_close          = -1;
    bool         allow_k2          = false; // K=2 is dominated on M3 unless masked-4 enables parity
    bool         force_k0_in_think = false;

    // Stage S8: Asynchronous Host-GPU Speculative Pipelining parameters
    bool         pipelined         = false; // overlap drafting of cycle N+1 with verification of cycle N
    double       eta               = 0.34;  // fraction of t_d hidden behind t_v (physical M3: (12.86ms - 4.5ms) / 24.5ms)
    double       t_sync_ms         = 3.0;   // ms, per-cycle event/condvar/handoff overhead
    double       g0                = 0.85;  // prior P(anchor guess == bonus token | all K accepted)

    // Stage S8 / P1: lookahead gating (formal lookahead elimination policy)
    double       tau_ms            = 14.7;  // GPU tax of expanding the DFlash batch K -> K+1 (calibrated, Condition B - A)
    double       cont_ms           = 2.0;   // extra target slowdown per failed overlap attempt (bandwidth/ALU contention)
};

struct snapshot_data {
    bool                 is_active = false;
    float                t_d0      = 1.5f;
    float                t_d       = 0.0f;
    float                slack13   = 0.0f;
    float                lcb       = 0.0f;
    float                beta_t    = 0.92f;
    std::array<float, 3> c         = {0.0f, 0.0f, 0.0f};
    std::array<float, 4> tps       = {0.0f, 0.0f, 0.0f, 0.0f};
};

class arbiter {
public:
    explicit arbiter(const arbiter_params & params = arbiter_params()) : P_(params), op_mode_(params.op_mode) {
        reset(true);
    }

    void reset(bool clear_history = false) {
        if (clear_history) {
            for (int j = 0; j <= arbiter_k_max; ++j) {
                a_[j] = 1.0;  // optimistic prior
                c_[j] = 1.0;
                n_obs_[j] = 0;
                last_measured_cycle_ms_[j] = P_.C[j];
            }
            g_            = std::clamp(P_.g0, 0.0, 1.0);
            total_cycles_ = 0;
        }
        k_cur_        = std::clamp(P_.baseline_k, 0, arbiter_k_max);
        n_warm_       = 0;
        park_cycles_  = 0;
        iv_           = P_.probe_iv_min;
        bail_counter_ = 0;
        probing_      = false;
        think_phase_  = false;
    }

    void set_mode(arbiter_mode m) { op_mode_ = m; }
    arbiter_mode get_mode() const { return op_mode_; }

    // ACTIVE: policy drives K. AUTO: policy drives K once warmup is complete.
    // SHADOW: policy is evaluated but baseline_k is executed.
    bool is_active() const {
        return op_mode_ == arbiter_mode::ACTIVE ||
              (op_mode_ == arbiter_mode::AUTO && n_warm_ >= P_.warmup);
    }

    bool in_think() const { return think_phase_; }

    // Feeds tokens to detect <think> and </think> delimiters
    void feed_token_id(int token_id) {
        if (P_.id_open >= 0 && token_id == P_.id_open) {
            think_phase_ = true;
        } else if (P_.id_close >= 0 && token_id == P_.id_close) {
            think_phase_ = false;
        } else if (token_id == 248068) {
            think_phase_ = true;
        } else if (token_id == 248069) {
            think_phase_ = false;
        }
    }

    // K the policy wants for the next cycle (includes due probe)
    int pred_k() const {
        if (probe_due_()) {
            return std::clamp(P_.probe_k, 1, 2);
        }
        return k_cur_;
    }

    // K to execute for the next cycle
    int select_k() {
        if (!is_active()) {
            return std::clamp(P_.baseline_k, 0, arbiter_k_max);
        }
        if (probe_due_()) {
            probing_ = true;
            return std::clamp(P_.probe_k, 1, 2);
        }
        probing_ = false;
        return k_cur_;
    }

    // Const overload for inspection
    int select_k() const {
        return is_active() ? pred_k() : std::clamp(P_.baseline_k, 0, arbiter_k_max);
    }

    // Stage S8: Asynchronous Pipelining controls
    void   set_pipelined(bool on) noexcept { P_.pipelined = on; }
    bool   is_pipelined() const noexcept { return P_.pipelined; }
    void   report_anchor(bool matched) noexcept {
        g_ = (1.0 - P_.alpha) * g_ + P_.alpha * (matched ? 1.0 : 0.0);
    }
    double hit_prob(int k) const noexcept {
        if (k < 1) return 0.0;
        const size_t kk = (size_t) std::min(k, arbiter_k_max);
        return std::clamp(a_[kk] * g_, 0.0, 1.0);
    }
    double cost_seq_ms(int k) const noexcept {
        const size_t kk = (size_t) std::clamp(k, 0, arbiter_k_max);
        return (last_measured_cycle_ms_[kk] > 0.0) ? last_measured_cycle_ms_[kk] : P_.C[kk];
    }
    // Expected cycle cost with overlap (ms):
    // hit  (prob h): tv + th + t_sync + (1 - eta) * td
    // miss (1 - h) : tv + th + t_sync + (1 - eta) * td + td
    // => C = tv + th + t_sync + (1 - eta) * td + (1 - h) * td
    double cost_pipe_ms(int k) const noexcept {
        if (k < 1 || P_.drafter != drafter_kind::BLOCK) return cost_seq_ms(k);
        const size_t kk = (size_t) std::min(k, arbiter_k_max);
        const double h  = hit_prob(k);
        return tv_[kk] + th_[kk] + P_.t_sync_ms + (1.0 - P_.eta) * td_[kk] + (1.0 - h) * td_[kk];
    }
    double cost_eff_ms(int k) const noexcept {
        const double seq = cost_seq_ms(k);
        if (!P_.pipelined || k < 1) return seq;
        return std::min(seq, cost_pipe_ms(k));
    }
    bool use_pipeline(int k) const noexcept {
        return P_.pipelined && k >= 1 && P_.drafter == drafter_kind::BLOCK && cost_pipe_ms(k) < cost_seq_ms(k);
    }

    // P1 -- value (ms) of paying the lookahead tax tau on a draft that runs on the critical path.
    // The lookahead token is the only source of the anchor guess, so paying tau opens an overlap chain:
    //   attempt i succeeds with prob h  -> saves s_hit = eta*td - t_sync, and the hidden worker draft carries the next
    //                                      lookahead for free (tau is hidden behind the remaining target layers)
    //   attempt i fails   with prob 1-h -> costs c_miss = t_sync + cont; the redraft is sequential, i.e. what the
    //                                      non-pipelined cycle would have paid anyway; the chain ends
    //   V(k) = -tau + sum_{i>=1} h^(i-1) * [h*s_hit - (1-h)*c_miss] = -tau + (h*s_hit - (1-h)*c_miss) / (1-h)
    double lookahead_value_ms(int k) const noexcept {
        if (!P_.pipelined || k < 1 || P_.drafter != drafter_kind::BLOCK) {
            return -P_.tau_ms;
        }
        const size_t kk     = (size_t) std::min(k, arbiter_k_max);
        const double h      = std::min(hit_prob(k), 0.98);   // chain length is bounded: 1/(1-h) <= 50
        const double s_hit  = P_.eta * td_[kk] - P_.t_sync_ms;
        if (s_hit <= 0.0) {
            // Overlap cannot cover sync + latency; paying lookahead is strictly negative
            return -P_.tau_ms + s_hit;
        }
        const double c_miss = P_.t_sync_ms + P_.cont_ms;
        return -P_.tau_ms + (h * s_hit - (1.0 - h) * c_miss) / (1.0 - h);
    }
    bool lookahead_worth(int k) const noexcept {
        return lookahead_value_ms(k) > P_.margin * P_.tau_ms;
    }
    double lookahead_tau_ms() const noexcept { return P_.tau_ms; }
    double anchor_rate() const noexcept { return g_; }

    // accepted_positions[i] != 0 <=> draft position i+1 was accepted by target
    void update(int k_executed, const std::vector<int> & accepted_positions) {
        const int k = std::clamp(k_executed, 0, arbiter_k_max);
        int n_acc = 0;
        while (n_acc < k && n_acc < (int) accepted_positions.size() && accepted_positions[n_acc] != 0) {
            ++n_acc;
        }
        update_prefix(k_executed, n_acc);
    }

    // Direct accepted-prefix length update
    void update_prefix(int k_executed, int n_accepted) {
        const int k = std::clamp(k_executed, 0, arbiter_k_max);
        total_cycles_++;

        if (k == 0) {
            if (k_cur_ == 0 && park_cycles_ < UINT32_MAX) {
                ++park_cycles_;
            }
            return;
        }

        n_accepted = std::clamp(n_accepted, 0, k);
        double s[arbiter_k_max + 1] = { 1.0, 0.0, 0.0, 0.0 };
        for (int j = 1; j <= k; ++j) {
            s[j] = (n_accepted >= j) ? 1.0 : 0.0;
        }

        // Handle probe return (Point 8 & 9 resolved: explicit state + hysteresis margin)
        if (probing_ || probe_due_()) {
            probing_     = false;
            park_cycles_ = 0;

            // Update conditional acceptance c_[1] via standard smoothed EMA (not raw overwrite)
            if (n_obs_[1] == 0) {
                c_[1] = s[1];
            } else {
                c_[1] += P_.alpha * (s[1] - c_[1]);
            }
            n_obs_[1]++;
            recompute_prefix_();

            // A probe succeeds if expected TPS at K=1 beats AR baseline K=0 with margin
            const double probe_t1 = (1.0 + a_[1]) / cost_eff_ms(1);
            const double t0_gate  = (1.0 / cost_eff_ms(0)) * (1.0 + P_.margin);

            if (probe_t1 > t0_gate) {
                // Probe succeeded: re-promote to K=1, reset backoff
                set_k_(1);
                n_warm_       = 0;
                iv_           = P_.probe_iv_min;
                bail_counter_ = 0;
            } else {
                // Probe failed: exponential backoff
                iv_ = std::min<uint32_t>(iv_ * 2u, P_.probe_iv_max);
            }
            return;
        }

        // Standard online conditional update (Point 5 resolved)
        // Position j was evaluated by target model iff position j-1 was accepted
        for (int j = 1; j <= k; ++j) {
            if (s[j - 1] > 0.5) {
                c_[j] += P_.alpha * (s[j] - c_[j]);
                n_obs_[j]++;
            }
        }
        recompute_prefix_();

        if (n_warm_ < P_.warmup) {
            ++n_warm_;
        }

        decide_();
    }

    // Compatibility shim for observe(...) call sites in speculative.cpp (Point 12)
    void observe(int k_drafted, int n_accepted, float draft_ms = 0.0f, float H_cycle = 0.0f) {
        if (k_drafted >= 1 && k_drafted <= arbiter_k_max) {
            if (draft_ms > 0.0f && td_ref_[k_drafted] > 0.0) {
                const double scale = std::clamp((double) draft_ms / td_ref_[k_drafted], 0.20, 3.00);
                for (int j = 1; j <= arbiter_k_max; ++j) {
                    const double target_td = (j == k_drafted) ? (double) draft_ms : (td_ref_[j] * scale);
                    td_[j] = (1.0 - P_.alpha) * td_[j] + P_.alpha * target_td;
                    last_measured_cycle_ms_[j] = tv_[j] + td_[j] + th_[j];
                }
            }
            if (H_cycle > 0.0f) {
                last_measured_cycle_ms_[k_drafted] = (1.0 - P_.alpha) * last_measured_cycle_ms_[k_drafted] + P_.alpha * (double) H_cycle;
            }
        }
        update_prefix(k_drafted, n_accepted);
    }

    // Snapshot telemetry for logging in speculative.cpp
    snapshot_data snapshot(mode /*m*/ = mode::content) const {
        snapshot_data snap;
        snap.is_active = is_active();
        snap.t_d0      = 1.5f;
        snap.t_d       = 0.0f;
        snap.beta_t    = 0.92f;

        for (int j = 0; j < 3; ++j) {
            snap.c[j] = (float) c_[j + 1];
        }

        const auto tps_all = compute_tps_();
        for (int j = 0; j <= arbiter_k_max; ++j) {
            snap.tps[j] = (float) tps_all[j];
        }

        snap.slack13 = (float) (tps_all[3] - tps_all[1]);
        snap.lcb     = (float) a_[1];
        return snap;
    }

private:
    double score_(int k) const {
        double sum = 0.0;
        for (int j = 1; j <= k; ++j) {
            sum += a_[j];
        }
        return sum;
    }

    bool probe_due_() const {
        return k_cur_ == 0 && park_cycles_ >= iv_;
    }

    // Strictly computes joint prefix probabilities algebraically (a_j = a_{j-1} * c_j)
    // Eliminates arbitrary min clipping and information distortion (Point 5 resolved)
    void recompute_prefix_() {
        a_[0] = 1.0;
        for (int j = 1; j <= arbiter_k_max; ++j) {
            a_[j] = a_[j - 1] * c_[j];
        }
    }

    // Direct expected throughput calculation with unobserved uncertainty gating (Points 6, 7, 20)
    std::array<double, arbiter_k_max + 1> compute_tps_() const {
        std::array<double, arbiter_k_max + 1> tps = {};
        tps[0] = 1000.0 / cost_eff_ms(0);
        for (int k = 1; k <= arbiter_k_max; ++k) {
            double eff_score = 0.0;
            double cur_a = 1.0;
            for (int j = 1; j <= k; ++j) {
                double eff_c = c_[j];
                // Progressive observation cap (GPT-5.6 Point 3 & 4 resolved):
                // Positions with fewer than warmup samples are gradually trusted rather than instant-unlocked
                if (n_obs_[j] < (uint32_t) P_.warmup) {
                    const double obs_cap = 0.90 + 0.0333 * (double) n_obs_[j];
                    eff_c = std::min(eff_c, obs_cap);
                }
                cur_a *= eff_c;
                eff_score += cur_a;
            }
            tps[k] = 1000.0 * (1.0 + eff_score) / cost_eff_ms(k);
        }
        return tps;
    }

    void set_k_(int k) {
        if (k == k_cur_) {
            return;
        }
        if (k == 0 || k_cur_ == 0) {
            park_cycles_ = 0;
            iv_          = P_.probe_iv_min;
        }
        k_cur_ = k;
    }

    // Direct expected throughput decision with asymmetric hysteresis
    void decide_() {
        if (n_warm_ < P_.warmup) {
            return;
        }

        if (P_.force_k0_in_think && think_phase_) {
            set_k_(0);
            return;
        }

        const auto   tps       = compute_tps_();
        const double cur_tps   = tps[k_cur_];
        const double up_margin = 1.0 + P_.margin;

        // 1. Fast Bailout Check to K=0:
        // If AR baseline beats current policy, check consecutive counter
        if (k_cur_ > 0 && tps[0] > cur_tps) {
            bail_counter_++;
            // Immediate bailout if severe collapse (a1 < 0.10) or consecutive confirmations
            if (a_[1] < 0.10 || bail_counter_ >= P_.consec_bail) {
                set_k_(0);
                bail_counter_ = 0;
                return;
            }
        } else {
            bail_counter_ = 0;
        }

        // 2. Promotion: search for candidate k > k_cur that beats current TPS by margin.
        //    In <think> phase, cap block drafter (DFlash2) at K=1 (M=2) so early warmup hits
        //    do not trigger M=4 (222 ms) verification on low-c2/c3 reasoning tokens, while
        //    ngram-mod still drafts up to K=3 at 0 ms cost.
        const int max_k_phase = (think_phase_ && P_.drafter == drafter_kind::BLOCK) ? 1 : arbiter_k_max;
        if (k_cur_ > max_k_phase) {
            set_k_(max_k_phase);
            return;
        }

        int best_promo_k = -1;
        double max_promo_tps = cur_tps * up_margin;

        for (int k = k_cur_ + 1; k <= max_k_phase; ++k) {
            if (!P_.allow_k2 && k == 2) {
                continue;
            }
            if (tps[k] > max_promo_tps) {
                max_promo_tps = tps[k];
                best_promo_k  = k;
            }
        }

        if (best_promo_k > 0) {
            set_k_(best_promo_k);
            return;
        }

        // 3. Demotion: search for candidate k < k_cur (k >= 1) that strictly beats current TPS.
        // Demotion to K=0 is strictly governed by Step 1 (consec_bail).
        int best_demo_k = -1;
        double max_demo_tps = cur_tps;

        for (int k = k_cur_ - 1; k >= 1; --k) {
            if (!P_.allow_k2 && k == 2) {
                continue;
            }
            if (tps[k] > max_demo_tps) {
                max_demo_tps = tps[k];
                best_demo_k  = k;
            }
        }

        if (best_demo_k >= 0) {
            set_k_(best_demo_k);
        }
    }

    arbiter_params P_;
    arbiter_mode   op_mode_      = arbiter_mode::ACTIVE;

    double   a_[arbiter_k_max + 1];  // EMA of cumulative prefix acceptance, a_[0] = 1.0
    double   c_[arbiter_k_max + 1];  // last known conditional ratios a_j / a_{j-1}
    uint32_t n_obs_[arbiter_k_max + 1]                  = {0};
    bool     probing_                                   = false;
    double   last_measured_cycle_ms_[arbiter_k_max + 1] = {0.0};
    int      k_cur_        = 1;
    int      n_warm_       = 0;
    uint32_t park_cycles_  = 0;
    uint32_t iv_           = 16;
    int      bail_counter_ = 0;
    uint64_t total_cycles_ = 0;
    bool     think_phase_  = false;

    // Stage S8 calibrated component latencies (Fused asym-Q2_K DFlash2 on M3, ms)
    std::array<double, arbiter_k_max + 1> tv_     = { 100.6, 130.3, 230.3, 222.0 }; // Target GPU
    std::array<double, arbiter_k_max + 1> td_     = {   0.0,  24.3,  39.7,  51.1 }; // Draft (DFlash proc + forward)
    std::array<double, arbiter_k_max + 1> td_ref_ = {   0.0,  24.3,  39.7,  51.1 }; // Reference Draft latency for scaling
    std::array<double, arbiter_k_max + 1> th_     = {   2.1,   2.2,   4.1,   4.6 }; // Host residual (C_K - tv - td)
    double                                g_  = 0.85; // online EMA of anchor match rate
};

}  // namespace spec
