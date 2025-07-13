#pragma once

#include "avx2_bhv.h"
#include <vector>
#include <map>

// Number of coarse clusters for the IVF-style index
constexpr int NUM_CLUSTERS = 256;

class SHMIndex {
public:
    SHMIndex();

    // Insert a BHV into the index
    void insert(const BHV& bhv);

    // Query the index for the k nearest neighbors
    std::vector<BHV> query(const BHV& query_bhv, int k);

private:
    // Centroids for the coarse clusters
    std::vector<BHV> centroids;

    // The inverted file: maps centroid index to a list of BHVs
    std::map<int, std::vector<BHV>> inverted_file;

    // Find the closest centroid to a given BHV
    int find_closest_centroid(const BHV& bhv);
};
