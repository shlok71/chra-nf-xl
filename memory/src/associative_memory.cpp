#include "associative_memory.h"
#include <limits>

AssociativeMemory::AssociativeMemory() {}

void AssociativeMemory::insert(const BHV& key, const BHV& value) {
    memory[key] = value;
}

BHV AssociativeMemory::query(const BHV& key) {
    int min_dist = std::numeric_limits<int>::max();
    BHV best_match;

    for (auto const& [mem_key, mem_value] : memory) {
        int dist = BHV::hamming_distance(key, mem_key);
        if (dist < min_dist) {
            min_dist = dist;
            best_match = mem_value;
        }
    }
    return best_match;
}

void AssociativeMemory::update(const BHV& key, const BHV& value) {
    // Hebbian-like update rule
    if (memory.count(key)) {
        BHV& existing_value = memory.at(key);
        for (int i = 0; i < BHV_WORDS; ++i) {
            existing_value.data[i] = (existing_value.data[i] & ~value.data[i]) | (value.data[i] & key.data[i]);
        }
    } else {
        memory[key] = value;
    }
}
