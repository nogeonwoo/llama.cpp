#pragma once
// Stage S8: Host-side hypothesis scheduler for asynchronous speculative pipelining.
// Pure C++17, slot-isolated, thread-safe condition variable synchronization.
//
// Target Hardware : Apple Silicon M3 (10 GPU cores, 24 GB Unified Memory)
// Concurrency Model: Worker thread handles drafter forward while Target GPU verifies Cycle N.

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace spec {

// Hypothesis H for cycle N+1, formed while cycle N is still being verified:
//   all `k_accept` drafts of cycle N are accepted AND the bonus token equals `anchor`.
struct hypothesis {
    uint64_t cycle    = 0;
    int32_t  anchor   = -1;   // guessed bonus token (= id_last of cycle N+1)
    int      k_accept = 0;    // == number of drafts verified in cycle N
};

struct draft_result {
    std::vector<int32_t> tokens;         // drafts for cycle N+1
    int32_t              lookahead = -1; // extra token beyond K, used as next anchor guess (-1: none)
    int64_t              t_us      = 0;  // worker wall time (feature ingest + draft)
    bool                 ok        = false;
};

using draft_fn = std::function<draft_result(const hypothesis &)>;

struct async_stats {
    uint64_t submitted = 0;
    uint64_t hits      = 0;
    uint64_t misses    = 0;   // submitted but hypothesis was wrong
    uint64_t skipped   = 0;   // resolve() with nothing submitted
};

class async_drafter {
public:
    explicit async_drafter(draft_fn fn) : fn_(std::move(fn)), worker_([this] { loop_(); }) {}

    async_drafter(const async_drafter &)             = delete;
    async_drafter & operator=(const async_drafter &) = delete;

    ~async_drafter() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_work_.notify_all();
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    // Non-blocking submission: safe to call from completion handlers or host thread
    bool submit(const hypothesis & h) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (st_ != state::idle) {
                return false;
            }
            hyp_ = h;
            st_  = state::pending;
            ++stats_.submitted;
        }
        cv_work_.notify_one();
        return true;
    }

    // Resolves the speculative draft against actual verification outcomes.
    // Blocks until worker finishes. If hypothesis matched, moves result to out.
    bool resolve(int32_t actual_anchor, int actual_accept, draft_result & out) {
        std::unique_lock<std::mutex> lk(mu_);
        if (st_ == state::idle) {
            ++stats_.skipped;
            return false;
        }
        cv_idle_.wait(lk, [this] { return st_ == state::done; });
        const bool hit = res_.ok && hyp_.anchor == actual_anchor && hyp_.k_accept == actual_accept;
        if (hit) {
            out = std::move(res_);
            ++stats_.hits;
        } else {
            ++stats_.misses;
        }
        res_ = draft_result{};
        st_  = state::idle;
        return hit;
    }

    // Discards any in-flight speculative hypothesis without applying tokens
    void quiesce() {
        std::unique_lock<std::mutex> lk(mu_);
        if (st_ == state::idle) {
            return;
        }
        cv_idle_.wait(lk, [this] { return st_ == state::done; });
        res_ = draft_result{};
        st_  = state::idle;
    }

    bool in_flight() const {
        std::lock_guard<std::mutex> lk(mu_);
        return st_ != state::idle;
    }

    async_stats stats() const {
        std::lock_guard<std::mutex> lk(mu_);
        return stats_;
    }

private:
    enum class state { idle, pending, running, done };

    void loop_() {
        std::unique_lock<std::mutex> lk(mu_);
        for (;;) {
            cv_work_.wait(lk, [this] { return stop_ || st_ == state::pending; });
            if (stop_) {
                return;
            }
            st_ = state::running;
            const hypothesis h = hyp_;
            lk.unlock();
            draft_result r = fn_(h);
            lk.lock();
            res_ = std::move(r);
            st_  = state::done;
            cv_idle_.notify_all();
        }
    }

    draft_fn                fn_;
    mutable std::mutex      mu_;
    std::condition_variable cv_work_;
    std::condition_variable cv_idle_;
    state                   st_   = state::idle;
    bool                    stop_ = false;
    hypothesis              hyp_{};
    draft_result            res_{};
    async_stats             stats_{};
    std::thread             worker_;
};

} // namespace spec
