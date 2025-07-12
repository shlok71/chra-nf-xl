#include "shm_index.h"
#include <algorithm>
#include <limits>

SHMIndex::SHMIndex() {
    // Initialize centroids with random BHVs.
    // In a real system, these would be learned (e.g., via K-means).
    for (int i = 0; i < NUM_CLUSTERS; ++i) {
        centroids.push_back(BHV::encode_text({"centroid", std::to_string(i)}));
    }
}

void SHMIndex::insert(const BHV& bhv) {
    int centroid_idx = find_closest_centroid(bhv);
    inverted_file[centroid_idx].push_back(bhv);
}

std::vector<BHV> SHMIndex::query(const BHV& query_bhv, int k) {
    int centroid_idx = find_closest_centroid(query_bhv);

    // For simplicity, we only search within the closest cluster.
    // A more advanced implementation would search neighboring clusters too (multi-probe).
    const auto& candidates = inverted_file[centroid_idx];

    // Sort candidates by Hamming distance to the query BHV
    std::vector<std::pair<int, BHV>> sorted_candidates;
    for (const auto& candidate : candidates) {
        int dist = BHV::hamming_distance(query_bhv, candidate);
        sorted_candidates.push_back({dist, candidate});
    }

    std::sort(sorted_candidates.begin(), sorted_candidates.end(),
              [](const auto& a, const auto& b) {
                  return a.first < b.first;
              });

    // Return the top k results
    std::vector<BHV> results;
    for (int i = 0; i < std::min((int)sorted_candidates.size(), k); ++i) {
        results.push_back(sorted_candidates[i].second);
    }
    return results;
}

int SHMIndex::find_closest_centroid(const BHV& bhv) {
    int min_dist = std::numeric_limits<int>::max();
    int closest_idx = -1;
    for (int i = 0; i < centroids.size(); ++i) {
        int dist = BHV::hamming_distance(bhv, centroids[i]);
        if (dist < min_dist) {
            min_dist = dist;
            closest_idx = i;
        }
    }
    return closest_idx;
}
