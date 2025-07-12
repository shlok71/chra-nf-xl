#include "avx2_bhv.h"
#include <stdexcept>
#include <functional> // For std::hash

// --- Constructors ---

BHV::BHV() {
    std::fill(data, data + BHV_WORDS, 0);
}

BHV::BHV(const BHV& other) {
    std::copy(other.data, other.data + BHV_WORDS, data);
}

BHV& BHV::operator=(const BHV& other) {
    if (this != &other) {
        std::copy(other.data, other.data + BHV_WORDS, data);
    }
    return *this;
}

// --- Core BHV Operations ---

/**
 * @brief Binds two BHV vectors using XOR operation.
 * Utilizes AVX2 for parallel XORing of 256-bit chunks.
 */
BHV BHV::bind(const BHV& a, const BHV& b) {
    BHV result;
    for (int i = 0; i < BHV_WORDS; i += 4) {
        __m256i vec_a = _mm256_load_si256((__m256i*)&a.data[i]);
        __m256i vec_b = _mm256_load_si256((__m256i*)&b.data[i]);
        __m256i vec_res = _mm256_xor_si256(vec_a, vec_b);
        _mm256_store_si256((__m256i*)&result.data[i], vec_res);
    }
    return result;
}

/**
 * @brief Computes the Hamming distance between two BHV vectors.
 * Uses AVX2's POPCNT instruction for efficient bit counting.
 */
int BHV::hamming_distance(const BHV& a, const BHV& b) {
    int distance = 0;
    __m256i chunk_a, chunk_b, xor_result;
    for (int i = 0; i < BHV_WORDS; i += 4) {
        chunk_a = _mm256_load_si256((__m256i*)&a.data[i]);
        chunk_b = _mm256_load_si256((__m256i*)&b.data[i]);
        xor_result = _mm256_xor_si256(chunk_a, chunk_b);

        // Unfortunately, AVX2 does not have a direct 256-bit popcount instruction.
        // We do it on 64-bit integers.
        uint64_t* C = (uint64_t*)&xor_result;
        for(int j=0; j<4; ++j) {
            distance += _mm_popcnt_u64(C[j]);
        }
    }
    return distance;
}

/**
 * @brief Encodes a vector of string tokens into a single BHV.
 * This is a simplified placeholder. A real implementation would use a robust
 * hashing scheme to map tokens to high-dimensional sparse binary vectors,
 * which are then combined. Here, we just XOR hash values.
 */
BHV BHV::encode_text(const std::vector<std::string>& tokens) {
    BHV result;
    std::hash<std::string> hasher;
    for (const auto& token : tokens) {
        size_t h = hasher(token);
        // Distribute the hash across the BHV vector.
        // This is a simplistic approach for demonstration.
        for (int i = 0; i < BHV_WORDS; ++i) {
            result.data[i] ^= (h << (i % 5)) | (h >> (64 - (i % 5)));
        }
    }
    return result;
}
