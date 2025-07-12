#!/bin/bash
# BVH Library Benchmark Script (Linux/macOS)

set -e

# --- Build the benchmark executable ---
# This assumes the main project build has already been run.
# A more robust script might have its own build step.
BENCH_EXE_NAME="bvh_benchmark"
BENCH_SRC_DIR="$(dirname "$0")"
BUILD_DIR="${BENCH_SRC_DIR}/../../build"
BENCH_EXE_PATH="${BUILD_DIR}/${BENCH_EXE_NAME}"

# --- Create a simple C++ benchmark file ---
cat > "${BUILD_DIR}/benchmark.cpp" << EOL
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include "avx2_bhv.h"

int main() {
    // --- Benchmark Parameters ---
    const int NUM_TOKENS = 1000;
    const int NUM_ITERATIONS = 100;

    std::vector<std::string> tokens;
    for (int i = 0; i < NUM_TOKENS; ++i) {
        tokens.push_back("token" + std::to_string(i));
    }

    // --- Benchmark: encode_text ---
    auto start_encode = std::chrono::high_resolution_clock::now();
    BHV encoded_bhv;
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        encoded_bhv = BHV::encode_text(tokens);
    }
    auto end_encode = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> encode_duration = end_encode - start_encode;
    double avg_encode_latency = encode_duration.count() / (NUM_ITERATIONS * NUM_TOKENS);

    // --- Benchmark: hamming_distance ---
    BHV another_bhv = BHV::encode_text({"another", "set", "of", "tokens"});
    auto start_hamming = std::chrono::high_resolution_clock::now();
    int distance;
    for (int i = 0; i < NUM_ITERATIONS * 100; ++i) {
        distance = BHV::hamming_distance(encoded_bhv, another_bhv);
    }
    auto end_hamming = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> hamming_duration = end_hamming - start_hamming;
    double avg_hamming_latency = hamming_duration.count() / (NUM_ITERATIONS * 100);

    // --- Output Results ---
    std::cout << "--- BHV Benchmark Results ---" << std::endl;
    std::cout << "Encode Latency per Token: " << avg_encode_latency << " ms" << std::endl;
    std::cout << "Hamming Distance Latency: " << avg_hamming_latency << " ms" << std::endl;

    if (avg_encode_latency > 1.0) {
        std::cerr << "Encode latency exceeds 1 ms/token!" << std::endl;
        return 1;
    }

    return 0;
}
EOL

# --- Compile the benchmark executable ---
echo "Compiling benchmark executable..."
g++ -std=c++17 -O3 -mavx2 -I"${BENCH_SRC_DIR}/../include" -L"${BUILD_DIR}/lib" \
    "${BUILD_DIR}/benchmark.cpp" -o "${BENCH_EXE_PATH}" -lbvh

# --- Run the benchmark ---
echo "Running benchmark..."
"${BENCH_EXE_PATH}"
