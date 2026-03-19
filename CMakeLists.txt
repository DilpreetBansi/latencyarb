cmake_minimum_required(VERSION 3.16)
project(LatencyArb VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -O3 -march=native")

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/src/cpp/include)

# Source files
set(LATENCYARB_SOURCES
    src/cpp/src/spsc_queue.cpp
    src/cpp/src/order_book.cpp
    src/cpp/src/execution_engine.cpp
)

# Header-only library
add_library(latencyarb_headers INTERFACE)
target_include_directories(latencyarb_headers INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(latencyarb_headers INTERFACE cxx_std_17)

# Optional: Create a static library target
add_library(latencyarb STATIC ${LATENCYARB_SOURCES})
target_link_libraries(latencyarb latencyarb_headers)
target_compile_options(latencyarb PRIVATE -pthread)

# Testing support
enable_testing()

# Unit test executable (optional)
add_executable(test_execution_engine_demo
    ${LATENCYARB_SOURCES}
)
target_link_libraries(test_execution_engine_demo latencyarb_headers)

# Installation rules
install(DIRECTORY ${CMAKE_SOURCE_DIR}/src/cpp/include/
        DESTINATION include/latencyarb)

install(TARGETS latencyarb
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
