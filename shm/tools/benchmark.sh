#!/bin/bash
# SHM Library Benchmark Script (Linux/macOS)

set -e

# --- Build the benchmark executable ---
BENCH_EXE_NAME="shm_benchmark"
BENCH_SRC_DIR="$(dirname "$0")"
BUILD_DIR="${BENCH_SRC_DIR}/../../build"
BENCH_EXE_PATH="${BUILD_DIR}/${BENCH_EXE_NAME}"

# --- Create a simple C++ benchmark file ---
cat > "${BUILD_DIR}/shm_benchmark.cpp" << EOL
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include "shm_index.h"

int main() {
    // --- Benchmark Parameters ---
    const int NUM_BHVS = 1000;
    const int NUM_QUERIES = 100;
    const int K = 10;

    SHMIndex index;
    std::vector<BHV> bhvs;
    for (int i = 0; i < NUM_BHVS; ++i) {
        bhvs.push_back(BHV::encode_text({"bhv" + std::to_string(i)}));
        index.insert(bhvs.back());
    }

    // --- Benchmark: query ---
    auto start_query = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_QUERIES; ++i) {
        index.query(bhvs[i], K);
    }
    auto end_query = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> query_duration = end_query - start_query;
    double avg_query_latency = query_duration.count() / NUM_QUERIES;

    // --- Output Results ---
    std::cout << "--- SHM Benchmark Results ---" << std::endl;
    std::cout << "Average Query Latency: " << avg_query_latency << " ms" << std::endl;

    if (avg_query_latency > 1.0) {
        std::cerr << "Query latency exceeds 1 ms!" << std::endl;
        return 1;
    }

    return 0;
}
EOL

# --- Compile the benchmark executable ---
echo "Compiling benchmark executable..."
g++ -std=c++17 -O3 -mavx2 -I"${BENCH_SRC_DIR}/../include" -I"${BENCH_SRC_DIR}/../../bvh/include" -L"${BUILD_DIR}/lib" \
    "${BUILD_DIR}/shm_benchmark.cpp" -o "${BENCH_EXE_PATH}" -lshm -lbvh

# --- Run the benchmark ---
echo "Running benchmark..."
"${BENCH_EXE_PATH}"
