# BVH Library Benchmark Script (Windows)

$ErrorActionPreference = "Stop"

# --- Build the benchmark executable ---
$BenchExeName = "bvh_benchmark.exe"
$BenchSrcDir = $PSScriptRoot
$BuildDir = Join-Path $BenchSrcDir "..\" "..\" "build"
$BenchExePath = Join-Path $BuildDir $BenchExeName

# --- Create a simple C++ benchmark file ---
$BenchmarkSource = @"
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
"@
Set-Content -Path (Join-Path $BuildDir "benchmark.cpp") -Value $BenchmarkSource

# --- Compile the benchmark executable ---
Write-Host "Compiling benchmark executable..."
$IncludeDir = Join-Path $BenchSrcDir "..\" "include"
$LibDir = Join-Path $BuildDir "lib"
cl.exe /EHsc /O2 /I $IncludeDir /Fe:$BenchExePath (Join-Path $BuildDir "benchmark.cpp") /link /LIBPATH:$LibDir bvh.lib

# --- Run the benchmark ---
Write-Host "Running benchmark..."
& $BenchExePath
