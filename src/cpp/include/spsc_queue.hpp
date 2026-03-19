#pragma once

#include <atomic>
#include <cstring>
#include <cassert>
#include <memory>

namespace latencyarb {

/**
 * Lock-free Single-Producer Single-Consumer queue using a ring buffer.
 * Thread-safe only when one thread produces and one thread consumes.
 *
 * Template parameters:
 * - T: element type (must be trivially copyable for best performance)
 * - CAPACITY: queue capacity (must be power of 2)
 *
 * Performance characteristics:
 * - O(1) amortized push/pop
 * - Zero allocations on hot path (pre-allocated ring buffer)
 * - Lock-free using atomic operations with acquire/release semantics
 */
template <typename T, size_t CAPACITY = 8192>
class SPSCQueue {
    static_assert((CAPACITY & (CAPACITY - 1)) == 0, "CAPACITY must be power of 2");
    static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");

public:
    SPSCQueue() : write_pos_(0), read_pos_(0) {
        // Pre-allocate entire buffer
        buffer_ = std::make_unique<T[]>(CAPACITY);
    }

    ~SPSCQueue() = default;

    // Non-copyable, non-movable
    SPSCQueue(const SPSCQueue&) = delete;
    SPSCQueue& operator=(const SPSCQueue&) = delete;
    SPSCQueue(SPSCQueue&&) = delete;
    SPSCQueue& operator=(SPSCQueue&&) = delete;

    /**
     * Try to push an element onto the queue.
     * @return true if successfully enqueued, false if queue is full
     */
    bool try_push(const T& value) noexcept {
        const size_t current_write = write_pos_.load(std::memory_order_relaxed);
        const size_t next_write = (current_write + 1) & (CAPACITY - 1);
        const size_t current_read = read_pos_.load(std::memory_order_acquire);

        if (next_write == current_read) {
            // Queue is full
            return false;
        }

        buffer_[current_write] = value;
        write_pos_.store(next_write, std::memory_order_release);
        return true;
    }

    /**
     * Try to pop an element from the queue.
     * @param[out] value: output parameter for the dequeued element
     * @return true if element was dequeued, false if queue is empty
     */
    bool try_pop(T& value) noexcept {
        const size_t current_read = read_pos_.load(std::memory_order_relaxed);
        const size_t current_write = write_pos_.load(std::memory_order_acquire);

        if (current_read == current_write) {
            // Queue is empty
            return false;
        }

        value = buffer_[current_read];
        read_pos_.store((current_read + 1) & (CAPACITY - 1), std::memory_order_release);
        return true;
    }

    /**
     * Push element (blocking until space available).
     * Spins if queue is full.
     */
    void push(const T& value) noexcept {
        while (!try_push(value)) {
            // Spin - in practice, a pause instruction or yield here
            __builtin_ia32_pause();
        }
    }

    /**
     * Pop element (blocking until element available).
     * Spins if queue is empty.
     */
    T pop() noexcept {
        T value;
        while (!try_pop(value)) {
            __builtin_ia32_pause();
        }
        return value;
    }

    /**
     * Get current queue size.
     * Note: may race with push/pop - only approximate.
     */
    size_t size() const noexcept {
        const size_t w = write_pos_.load(std::memory_order_acquire);
        const size_t r = read_pos_.load(std::memory_order_acquire);
        return (w - r) & (CAPACITY - 1);
    }

    /**
     * Check if queue is empty.
     */
    bool empty() const noexcept {
        return read_pos_.load(std::memory_order_acquire) ==
               write_pos_.load(std::memory_order_acquire);
    }

    /**
     * Check if queue is full.
     */
    bool full() const noexcept {
        const size_t w = write_pos_.load(std::memory_order_acquire);
        const size_t r = read_pos_.load(std::memory_order_acquire);
        return ((w + 1) & (CAPACITY - 1)) == r;
    }

    /**
     * Get queue capacity.
     */
    static constexpr size_t capacity() noexcept {
        return CAPACITY;
    }

    /**
     * Clear the queue (single-threaded only).
     */
    void clear() noexcept {
        write_pos_.store(0, std::memory_order_release);
        read_pos_.store(0, std::memory_order_release);
    }

private:
    // Cache-line aligned to avoid false sharing
    alignas(64) std::atomic<size_t> write_pos_;
    alignas(64) std::atomic<size_t> read_pos_;
    std::unique_ptr<T[]> buffer_;
};

}  // namespace latencyarb
