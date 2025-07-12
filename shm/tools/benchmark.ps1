# SHM Library Benchmark Script (Windows)

$ErrorActionPreference = "Stop"

# --- Build the benchmark executable ---
$BenchExeName = "shm_benchmark.exe"
$BenchSrcDir = $PSScriptRoot
$BuildDir = Join-Path $BenchSrcDir "..\" "..\" "build"
$BenchExePath = Join-Path $BuildDir $BenchExeName

# --- Create a simple C++ benchmark file ---
$BenchmarkSource = @"
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
"@
Set-Content -Path (Join-Path $BuildDir "shm_benchmark.cpp") -Value $BenchmarkSource

# --- Compile the benchmark executable ---
Write-Host "Compiling benchmark executable..."
$BvhIncludeDir = Join-Path $BenchSrcDir "..\" "..\" "bvh" "include"
$ShmIncludeDir = Join-Path $BenchSrcDir "..\" "include"
$LibDir = Join-Path $BuildDir "lib"
cl.exe /EHsc /O2 /I $BvhIncludeDir /I $ShmIncludeDir /Fe:$BenchExePath (Join-Path $BuildDir "shm_benchmark.cpp") /link /LIBPATH:$LibDir shm.lib bvh.lib

# --- Run the benchmark ---
Write-Host "Running benchmark..."
& $BenchExePath
