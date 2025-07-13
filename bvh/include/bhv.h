#pragma once

#include <vector>
#include <string>
#include <cstdint>

// Define the size of the BHV vector in bits and 64-bit words
constexpr int BHV_BITS = 16384;
constexpr int BHV_WORDS = BHV_BITS / 64;

// BHV class representing a 16,384-bit vector
class BHV {
public:
    // Aligned storage for the BHV data
    alignas(32) uint64_t data[BHV_WORDS];

    // Constructors
    BHV();
    BHV(const BHV& other);

    // Operators
    BHV& operator=(const BHV& other);

    // Static methods for BHV operations
    static BHV encode(const std::string& input);
    static std::string decode(const BHV& bhv);
    static BHV bind(const BHV& a, const BHV& b);
    static BHV bundle(const std::vector<BHV>& bhvs);
};
