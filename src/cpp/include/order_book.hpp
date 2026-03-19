#pragma once

#include "types.hpp"
#include <map>
#include <memory>
#include <cmath>

namespace latencyarb {

/**
 * L2 Order Book with sorted price levels.
 * Tracks bid and ask side separately with O(log N) updates.
 *
 * Designed for:
 * - Fast best bid/ask queries
 * - Mid-price and spread calculations
 * - Volume-at-level queries
 * - Realistic market microstructure simulation
 */
class OrderBook {
public:
    OrderBook(uint32_t symbol_id)
        : symbol_id_(symbol_id),
          best_bid_(-1.0),
          best_ask_(std::numeric_limits<double>::max()),
          bid_volume_(0),
          ask_volume_(0),
          last_update_time_(0) {}

    ~OrderBook() = default;

    // Non-copyable
    OrderBook(const OrderBook&) = delete;
    OrderBook& operator=(const OrderBook&) = delete;

    /**
     * Apply a market data update to the order book.
     * @param update: BookUpdate containing bid/ask prices and sizes
     */
    void apply_update(const BookUpdate& update) noexcept {
        if (update.symbol_id != symbol_id_) {
            return;
        }

        best_bid_ = update.bid_price;
        best_ask_ = update.ask_price;
        bid_volume_ = update.bid_size;
        ask_volume_ = update.ask_size;
        last_update_time_ = update.timestamp;

        // Update bid side
        if (update.bid_price > 0.0 && update.bid_size > 0) {
            bids_[update.bid_price] = update.bid_size;
        } else if (update.bid_price > 0.0) {
            bids_.erase(update.bid_price);
        }

        // Update ask side
        if (update.ask_price > 0.0 && update.ask_size > 0) {
            asks_[update.ask_price] = update.ask_size;
        } else if (update.ask_price > 0.0) {
            asks_.erase(update.ask_price);
        }
    }

    /**
     * Get best bid price.
     */
    double best_bid() const noexcept {
        return best_bid_;
    }

    /**
     * Get best ask price.
     */
    double best_ask() const noexcept {
        return best_ask_;
    }

    /**
     * Get mid price (average of bid and ask).
     */
    double mid_price() const noexcept {
        if (best_bid_ < 0.0 || best_ask_ >= std::numeric_limits<double>::max()) {
            return 0.0;
        }
        return (best_bid_ + best_ask_) / 2.0;
    }

    /**
     * Get spread (ask - bid).
     */
    double spread() const noexcept {
        if (best_bid_ < 0.0 || best_ask_ >= std::numeric_limits<double>::max()) {
            return 0.0;
        }
        return best_ask_ - best_bid_;
    }

    /**
     * Get spread in basis points.
     */
    double spread_bps() const noexcept {
        double mp = mid_price();
        if (mp == 0.0) {
            return 0.0;
        }
        return (spread() / mp) * 10000.0;
    }

    /**
     * Get volume at best bid.
     */
    int64_t bid_volume() const noexcept {
        return bid_volume_;
    }

    /**
     * Get volume at best ask.
     */
    int64_t ask_volume() const noexcept {
        return ask_volume_;
    }

    /**
     * Get total volume at a specific price level.
     * @return volume if level exists, 0 otherwise
     */
    int64_t volume_at_level(double price) const noexcept {
        // Check bids
        auto bid_it = bids_.find(price);
        if (bid_it != bids_.end()) {
            return bid_it->second;
        }

        // Check asks
        auto ask_it = asks_.find(price);
        if (ask_it != asks_.end()) {
            return ask_it->second;
        }

        return 0;
    }

    /**
     * Get total volume at N best levels on bid side.
     */
    int64_t cumulative_bid_volume(size_t levels) const noexcept {
        int64_t total = 0;
        size_t count = 0;

        // Iterate from highest bid downwards
        for (auto it = bids_.rbegin(); it != bids_.rend() && count < levels; ++it, ++count) {
            total += it->second;
        }

        return total;
    }

    /**
     * Get total volume at N best levels on ask side.
     */
    int64_t cumulative_ask_volume(size_t levels) const noexcept {
        int64_t total = 0;
        size_t count = 0;

        // Iterate from lowest ask upwards
        for (auto it = asks_.begin(); it != asks_.end() && count < levels; ++it, ++count) {
            total += it->second;
        }

        return total;
    }

    /**
     * Get symbol ID.
     */
    uint32_t symbol_id() const noexcept {
        return symbol_id_;
    }

    /**
     * Get last update timestamp.
     */
    int64_t last_update_time() const noexcept {
        return last_update_time_;
    }

    /**
     * Check if order book is valid (has both bid and ask).
     */
    bool is_valid() const noexcept {
        return best_bid_ > 0.0 && best_ask_ < std::numeric_limits<double>::max() &&
               best_bid_ < best_ask_;
    }

    /**
     * Clear the order book (single-threaded only).
     */
    void clear() noexcept {
        bids_.clear();
        asks_.clear();
        best_bid_ = -1.0;
        best_ask_ = std::numeric_limits<double>::max();
        bid_volume_ = 0;
        ask_volume_ = 0;
    }

    /**
     * Get depth snapshot at multiple levels.
     */
    struct Snapshot {
        std::vector<std::pair<double, int64_t>> bids;  // price, size pairs
        std::vector<std::pair<double, int64_t>> asks;
    };

    Snapshot get_snapshot(size_t levels = 10) const noexcept {
        Snapshot snap;

        // Top bids
        size_t count = 0;
        for (auto it = bids_.rbegin(); it != bids_.rend() && count < levels; ++it, ++count) {
            snap.bids.push_back({it->first, it->second});
        }

        // Top asks
        count = 0;
        for (auto it = asks_.begin(); it != asks_.end() && count < levels; ++it, ++count) {
            snap.asks.push_back({it->first, it->second});
        }

        return snap;
    }

private:
    uint32_t symbol_id_;
    double best_bid_;
    double best_ask_;
    int64_t bid_volume_;
    int64_t ask_volume_;
    int64_t last_update_time_;

    // Maps: price -> volume
    // std::map keeps prices sorted automatically
    std::map<double, int64_t> bids_;   // Descending order (rbegin gives best bid)
    std::map<double, int64_t> asks_;   // Ascending order (begin gives best ask)
};

}  // namespace latencyarb
