#include "bhv.h"
#include <functional> // For std::hash
#include <immintrin.h>

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

BHV BHV::encode(const std::string& input) {
    BHV result;
    std::hash<std::string> hasher;
    size_t h = hasher(input);
    for (int i = 0; i < BHV_WORDS; ++i) {
        result.data[i] = (h << (i % 5)) | (h >> (64 - (i % 5)));
    }
    return result;
}

std::string BHV::decode(const BHV& bhv) {
    // This is a placeholder for the decode function.
    // A real implementation would require a more sophisticated
    // decoding scheme.
    return "decoded_string";
}

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

BHV BHV::bundle(const std::vector<BHV>& bhvs) {
    BHV result;
    if (bhvs.empty()) {
        return result;
    }
    result = bhvs[0];
    for (size_t i = 1; i < bhvs.size(); ++i) {
        for (int j = 0; j < BHV_WORDS; ++j) {
            result.data[j] ^= bhvs[i].data[j];
        }
    }
    return result;
}
