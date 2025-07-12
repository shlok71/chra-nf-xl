#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <immintrin.h>

// Define the size of the BHV vector in bits and 64-bit words
constexpr int BHV_BITS = 16384;
constexpr int BHV_WORDS = BHV_BITS / 64;

// Alignment for AVX2 operations
#define ALIGNMENT 32

// BHV class representing a 16,384-bit vector
class BHV {
public:
    // Aligned storage for the BHV data
    alignas(ALIGNMENT) uint64_t data[BHV_WORDS];

    // Constructors
    BHV();
    BHV(const BHV& other);

    // Operators
    BHV& operator=(const BHV& other);

    // Static methods for BHV operations
    static BHV encode_text(const std::vector<std::string>& tokens);
    static BHV bind(const BHV& a, const BHV& b);
    static int hamming_distance(const BHV& a, const BHV& b);
};
