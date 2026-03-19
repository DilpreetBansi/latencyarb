#pragma once

#include "types.hpp"
#include "spsc_queue.hpp"
#include "order_book.hpp"
#include <unordered_map>
#include <memory>
#include <chrono>

namespace latencyarb {

/**
 * Strategy interface for execution engine.
 * Concrete strategies must implement this interface.
 */
class Strategy {
public:
    virtual ~Strategy() = default;

    /**
     * Called when new market data arrives.
     * Strategy can submit orders through engine.on_signal()
     */
    virtual void on_market_data(const BookUpdate& update) = 0;

    /**
     * Called when an order gets filled.
     */
    virtual void on_fill(const Fill& fill) = 0;

    /**
     * Get strategy ID.
     */
    virtual uint64_t strategy_id() const = 0;
};

/**
 * Ultra-low-latency execution engine.
 * Template-based to work with any Strategy implementation.
 *
 * Features:
 * - Lock-free order queue
 * - Position tracking per symbol
 * - PnL calculation with mark-to-market
 * - Order and fill tracking
 * - Callback-driven architecture
 */
class ExecutionEngine {
public:
    ExecutionEngine() : next_order_id_(1), next_fill_id_(1), total_cash_(0.0) {}

    ~ExecutionEngine() = default;

    /**
     * Initialize engine with starting capital.
     */
    void initialize(double initial_cash) noexcept {
        total_cash_ = initial_cash;
        starting_cash_ = initial_cash;
    }

    /**
     * Register a strategy with the engine.
     */
    void register_strategy(Strategy* strategy) noexcept {
        if (strategy) {
            strategies_.push_back(strategy);
        }
    }

    /**
     * Process market data update through all strategies.
     */
    void on_market_data(const BookUpdate& update) noexcept {
        // Update order book
        if (order_books_.find(update.symbol_id) == order_books_.end()) {
            order_books_[update.symbol_id] = std::make_unique<OrderBook>(update.symbol_id);
        }
        order_books_[update.symbol_id]->apply_update(update);

        // Call all strategies
        for (auto strategy : strategies_) {
            strategy->on_market_data(update);
        }
    }

    /**
     * Submit a signal/order from strategy.
     * Returns true if order was enqueued, false if queue full.
     */
    bool submit_order(uint64_t strategy_id, uint32_t symbol_id, Side side,
                      double price, int64_t quantity) noexcept {
        Order order(next_order_id_++, symbol_id, side, price, quantity);
        order.strategy_id = strategy_id;
        order.timestamp = get_timestamp();

        // Try to enqueue order
        return order_queue_.try_push(order);
    }

    /**
     * Process all pending orders in the queue.
     * Simulates order execution and generates fills.
     */
    void process_orders() noexcept {
        Order order;
        while (order_queue_.try_pop(order)) {
            execute_order(order);
        }
    }

    /**
     * Get best bid for symbol.
     */
    double get_best_bid(uint32_t symbol_id) const noexcept {
        auto it = order_books_.find(symbol_id);
        if (it != order_books_.end()) {
            return it->second->best_bid();
        }
        return 0.0;
    }

    /**
     * Get best ask for symbol.
     */
    double get_best_ask(uint32_t symbol_id) const noexcept {
        auto it = order_books_.find(symbol_id);
        if (it != order_books_.end()) {
            return it->second->best_ask();
        }
        return 0.0;
    }

    /**
     * Get mid price for symbol.
     */
    double get_mid_price(uint32_t symbol_id) const noexcept {
        auto it = order_books_.find(symbol_id);
        if (it != order_books_.end()) {
            return it->second->mid_price();
        }
        return 0.0;
    }

    /**
     * Get order book for symbol.
     */
    OrderBook* get_order_book(uint32_t symbol_id) noexcept {
        auto it = order_books_.find(symbol_id);
        if (it != order_books_.end()) {
            return it->second.get();
        }
        return nullptr;
    }

    /**
     * Get position for symbol.
     */
    Position get_position(uint32_t symbol_id) const noexcept {
        auto it = positions_.find(symbol_id);
        if (it != positions_.end()) {
            return it->second;
        }
        return Position();
    }

    /**
     * Get current cash balance.
     */
    double get_cash() const noexcept {
        return total_cash_;
    }

    /**
     * Get total portfolio value (cash + positions at mid price).
     */
    double get_portfolio_value() const noexcept {
        double value = total_cash_;
        for (const auto& [symbol_id, position] : positions_) {
            if (position.quantity != 0) {
                double mid_price = get_mid_price(symbol_id);
                if (mid_price > 0.0) {
                    value += position.quantity * mid_price;
                }
            }
        }
        return value;
    }

    /**
     * Get unrealized PnL for a position.
     */
    double get_unrealized_pnl(uint32_t symbol_id) const noexcept {
        auto it = positions_.find(symbol_id);
        if (it != positions_.end()) {
            return it->second.unrealized_pnl;
        }
        return 0.0;
    }

    /**
     * Get total unrealized PnL across all positions.
     */
    double get_total_unrealized_pnl() const noexcept {
        double total = 0.0;
        for (const auto& [symbol_id, position] : positions_) {
            total += position.unrealized_pnl;
        }
        return total;
    }

    /**
     * Get total realized PnL.
     */
    double get_total_realized_pnl() const noexcept {
        double total = 0.0;
        for (const auto& [symbol_id, position] : positions_) {
            total += position.realized_pnl;
        }
        return total;
    }

    /**
     * Update mark-to-market for all positions.
     */
    void update_mtm() noexcept {
        for (auto& [symbol_id, position] : positions_) {
            if (position.quantity != 0) {
                double mid_price = get_mid_price(symbol_id);
                if (mid_price > 0.0) {
                    position.unrealized_pnl =
                        position.quantity * (mid_price - position.average_entry_price);
                }
            }
        }
    }

    /**
     * Get number of filled orders.
     */
    size_t get_fill_count() const noexcept {
        return fills_.size();
    }

    /**
     * Get fill at index.
     */
    Fill get_fill(size_t index) const noexcept {
        if (index < fills_.size()) {
            return fills_[index];
        }
        return Fill();
    }

private:
    /**
     * Execute a single order (simulate fill).
     */
    void execute_order(Order& order) noexcept {
        auto book = get_order_book(order.symbol_id);
        if (!book || !book->is_valid()) {
            order.status = OrderStatus::REJECTED;
            return;
        }

        double execution_price;
        bool can_fill = false;

        if (order.side == Side::BUY) {
            execution_price = book->best_ask();
            can_fill = (order.price >= execution_price);
        } else {
            execution_price = book->best_bid();
            can_fill = (order.price <= execution_price);
        }

        if (!can_fill) {
            order.status = OrderStatus::REJECTED;
            return;
        }

        // Create fill
        Fill fill(next_fill_id_++, order.order_id, order.symbol_id, order.side,
                  execution_price, order.quantity);
        fill.timestamp = get_timestamp();
        fill.commission = 0.001 * execution_price * order.quantity;  // 0.1 bps

        // Update position
        update_position(order.symbol_id, order.side, order.quantity, execution_price);

        // Update cash
        double cash_impact = order.quantity * execution_price;
        if (order.side == Side::BUY) {
            total_cash_ -= (cash_impact + fill.commission);
        } else {
            total_cash_ += (cash_impact - fill.commission);
        }

        order.status = OrderStatus::FILLED;
        order.filled_quantity = order.quantity;
        order.average_fill_price = execution_price;
        order.fill_timestamp = fill.timestamp;

        fills_.push_back(fill);

        // Notify strategy of fill
        for (auto strategy : strategies_) {
            if (strategy->strategy_id() == order.strategy_id) {
                strategy->on_fill(fill);
            }
        }
    }

    /**
     * Update position after a fill.
     */
    void update_position(uint32_t symbol_id, Side side, int64_t quantity, double price) noexcept {
        Position& pos = positions_[symbol_id];

        if (pos.symbol_id < 0) {
            pos.symbol_id = symbol_id;
        }

        double cost = quantity * price;

        if (pos.quantity == 0) {
            // Opening position
            pos.quantity = (side == Side::BUY) ? quantity : -quantity;
            pos.average_entry_price = price;
        } else if ((pos.quantity > 0 && side == Side::BUY) ||
                   (pos.quantity < 0 && side == Side::SELL)) {
            // Adding to position
            double avg_cost = pos.average_entry_price * std::abs(pos.quantity);
            pos.average_entry_price = (avg_cost + cost) / (std::abs(pos.quantity) + quantity);
            pos.quantity += (side == Side::BUY) ? quantity : -quantity;
        } else {
            // Reducing or closing position
            int64_t close_qty = std::min(std::abs(pos.quantity), quantity);
            pos.realized_pnl += (side == Side::BUY) ? -close_qty * (price - pos.average_entry_price)
                                                     : close_qty * (price - pos.average_entry_price);

            pos.quantity += (side == Side::BUY) ? quantity : -quantity;

            if (pos.quantity != 0) {
                // Still some position left, average price unchanged
            } else {
                pos.average_entry_price = 0.0;
            }
        }

        update_mtm();
    }

    /**
     * Get current timestamp in nanoseconds.
     */
    static int64_t get_timestamp() noexcept {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    }

    // Configuration
    static constexpr size_t ORDER_QUEUE_SIZE = 4096;

    // State
    SPSCQueue<Order, ORDER_QUEUE_SIZE> order_queue_;
    std::vector<Strategy*> strategies_;
    std::unordered_map<uint32_t, std::unique_ptr<OrderBook>> order_books_;
    std::unordered_map<uint32_t, Position> positions_;
    std::vector<Fill> fills_;

    uint64_t next_order_id_;
    uint64_t next_fill_id_;
    double total_cash_;
    double starting_cash_;
};

}  // namespace latencyarb
