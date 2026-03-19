#pragma once

#include <cstdint>
#include <chrono>
#include <string>

namespace latencyarb {

// Order-related types
enum class Side : uint8_t {
    BUY = 0,
    SELL = 1
};

enum class OrderType : uint8_t {
    MARKET = 0,
    LIMIT = 1
};

enum class OrderStatus : uint8_t {
    PENDING = 0,
    FILLED = 1,
    CANCELLED = 2,
    REJECTED = 3
};

// Timestamp type: nanoseconds since epoch for ultra-low latency precision
using Timestamp = int64_t;

// Order structure - aligned for cache efficiency
struct Order {
    uint64_t order_id;
    uint64_t strategy_id;
    uint32_t symbol_id;
    Side side;
    OrderType type;
    OrderStatus status;
    uint8_t _pad1;  // padding for alignment

    double price;
    int64_t quantity;
    int64_t filled_quantity;
    double average_fill_price;

    Timestamp timestamp;
    Timestamp fill_timestamp;

    Order() = default;

    Order(uint64_t id, uint32_t sym_id, Side s, double p, int64_t qty)
        : order_id(id), strategy_id(0), symbol_id(sym_id), side(s),
          type(OrderType::LIMIT), status(OrderStatus::PENDING),
          _pad1(0), price(p), quantity(qty), filled_quantity(0),
          average_fill_price(0.0), timestamp(0), fill_timestamp(0) {}
};

// Fill structure - represents partial/full execution
struct Fill {
    uint64_t fill_id;
    uint64_t order_id;
    uint32_t symbol_id;
    Side side;
    uint8_t _pad1[3];

    double price;
    int64_t quantity;
    double commission;

    Timestamp timestamp;

    Fill() = default;

    Fill(uint64_t fid, uint64_t oid, uint32_t sym_id, Side s, double p, int64_t qty)
        : fill_id(fid), order_id(oid), symbol_id(sym_id), side(s),
          _pad1{0, 0, 0}, price(p), quantity(qty), commission(0.0), timestamp(0) {}
};

// Market data structures
struct BookUpdate {
    uint32_t symbol_id;
    uint8_t _pad1[4];

    double bid_price;
    double ask_price;
    int64_t bid_size;
    int64_t ask_size;

    Timestamp timestamp;

    BookUpdate() = default;

    BookUpdate(uint32_t sym_id, double bid, double ask, int64_t bsize, int64_t asize)
        : symbol_id(sym_id), _pad1{0, 0, 0, 0}, bid_price(bid), ask_price(ask),
          bid_size(bsize), ask_size(asize), timestamp(0) {}
};

// Price level for order book
struct PriceLevel {
    double price;
    int64_t quantity;

    PriceLevel() = default;
    PriceLevel(double p, int64_t q) : price(p), quantity(q) {}

    bool operator<(const PriceLevel& other) const {
        return price < other.price;
    }
};

// Position tracking
struct Position {
    int32_t symbol_id;
    int64_t quantity;
    double average_entry_price;
    double unrealized_pnl;
    double realized_pnl;

    Position() : symbol_id(-1), quantity(0), average_entry_price(0.0),
                 unrealized_pnl(0.0), realized_pnl(0.0) {}
};

}  // namespace latencyarb
