#pragma once

#include <memory>
#include <cstdint>
#include <cassert>

namespace latencyarb {

/**
 * Pre-allocated memory pool for fixed-size objects.
 * Provides O(1) allocation and deallocation using a free-list.
 * No system allocator calls on hot path.
 *
 * Template parameters:
 * - T: element type
 * - POOL_SIZE: maximum number of objects that can be allocated
 *
 * Thread safety: NOT thread-safe. Designed for single-threaded use.
 * For multi-threaded use, use one pool per thread or synchronize access.
 */
template <typename T, size_t POOL_SIZE = 4096>
class MemoryPool {
public:
    MemoryPool() : free_list_(nullptr), allocation_count_(0) {
        // Pre-allocate all objects
        buffer_ = std::make_unique<Node[]>(POOL_SIZE);

        // Initialize free list - link all nodes
        for (size_t i = 0; i < POOL_SIZE - 1; ++i) {
            buffer_[i].next = &buffer_[i + 1];
        }
        buffer_[POOL_SIZE - 1].next = nullptr;

        free_list_ = &buffer_[0];
    }

    ~MemoryPool() = default;

    // Non-copyable, non-movable
    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;
    MemoryPool(MemoryPool&&) = delete;
    MemoryPool& operator=(MemoryPool&&) = delete;

    /**
     * Allocate an object from the pool.
     * @return pointer to allocated object, nullptr if pool exhausted
     */
    T* allocate() noexcept {
        if (!free_list_) {
            return nullptr;  // Pool exhausted
        }

        Node* node = free_list_;
        free_list_ = node->next;
        allocation_count_++;

        // Construct object in-place
        return new (&node->storage) T();
    }

    /**
     * Allocate with constructor arguments.
     */
    template <typename... Args>
    T* allocate(Args&&... args) noexcept {
        if (!free_list_) {
            return nullptr;
        }

        Node* node = free_list_;
        free_list_ = node->next;
        allocation_count_++;

        return new (&node->storage) T(std::forward<Args>(args)...);
    }

    /**
     * Return an object to the pool.
     * @param ptr: pointer to object (must have been allocated from this pool)
     */
    void deallocate(T* ptr) noexcept {
        if (!ptr) {
            return;
        }

        // Call destructor
        ptr->~T();

        // Add node back to free list
        Node* node = reinterpret_cast<Node*>(ptr);
        node->next = free_list_;
        free_list_ = node;
        allocation_count_--;
    }

    /**
     * Get number of currently allocated objects.
     */
    size_t allocation_count() const noexcept {
        return allocation_count_;
    }

    /**
     * Get number of available objects in pool.
     */
    size_t available() const noexcept {
        return POOL_SIZE - allocation_count_;
    }

    /**
     * Reset pool to initial state (single-threaded only).
     */
    void clear() noexcept {
        allocation_count_ = 0;
        for (size_t i = 0; i < POOL_SIZE - 1; ++i) {
            buffer_[i].next = &buffer_[i + 1];
        }
        buffer_[POOL_SIZE - 1].next = nullptr;
        free_list_ = &buffer_[0];
    }

    /**
     * Get pool capacity.
     */
    static constexpr size_t capacity() noexcept {
        return POOL_SIZE;
    }

private:
    struct Node {
        alignas(T) char storage[sizeof(T)];
        Node* next;
    };

    std::unique_ptr<Node[]> buffer_;
    Node* free_list_;
    size_t allocation_count_;
};

/**
 * RAII wrapper for pool-allocated objects.
 * Automatically returns object to pool on destruction.
 */
template <typename T, size_t POOL_SIZE = 4096>
class PoolPtr {
public:
    PoolPtr(MemoryPool<T, POOL_SIZE>* pool, T* ptr)
        : pool_(pool), ptr_(ptr) {}

    ~PoolPtr() {
        if (pool_ && ptr_) {
            pool_->deallocate(ptr_);
        }
    }

    // Move semantics
    PoolPtr(PoolPtr&& other) noexcept
        : pool_(other.pool_), ptr_(other.ptr_) {
        other.pool_ = nullptr;
        other.ptr_ = nullptr;
    }

    PoolPtr& operator=(PoolPtr&& other) noexcept {
        if (this != &other) {
            if (pool_ && ptr_) {
                pool_->deallocate(ptr_);
            }
            pool_ = other.pool_;
            ptr_ = other.ptr_;
            other.pool_ = nullptr;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    // Non-copyable
    PoolPtr(const PoolPtr&) = delete;
    PoolPtr& operator=(const PoolPtr&) = delete;

    T* operator->() noexcept { return ptr_; }
    T& operator*() noexcept { return *ptr_; }
    const T* operator->() const noexcept { return ptr_; }
    const T& operator*() const noexcept { return *ptr_; }
    T* get() noexcept { return ptr_; }
    bool operator!=(nullptr_t) const noexcept { return ptr_ != nullptr; }

private:
    MemoryPool<T, POOL_SIZE>* pool_;
    T* ptr_;
};

}  // namespace latencyarb
