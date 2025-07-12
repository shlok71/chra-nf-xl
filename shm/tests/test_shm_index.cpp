#include "gtest/gtest.h"
#include "shm_index.h"
#include <vector>
#include <string>
#include <set>

// Helper function to generate a unique string from a BHV
std::string bhv_to_string(const BHV& bhv) {
    std::string s = "";
    for (int i = 0; i < BHV_WORDS; ++i) {
        s += std::to_string(bhv.data[i]);
    }
    return s;
}

TEST(SHMIndexTest, Recall) {
    const int NUM_BHVS = 1000;
    const int K = 10;

    SHMIndex index;
    std::vector<BHV> inserted_bhvs;

    // Insert 1000 random BHVs
    for (int i = 0; i < NUM_BHVS; ++i) {
        BHV bhv = BHV::encode_text({"random_bhv_" + std::to_string(i)});
        inserted_bhvs.push_back(bhv);
        index.insert(bhv);
    }

    int recall_at_10_count = 0;
    // Query for each inserted BHV
    for (const auto& query_bhv : inserted_bhvs) {
        std::vector<BHV> results = index.query(query_bhv, K);

        // Check if the original BHV is in the results
        bool found = false;
        for (const auto& result_bhv : results) {
            if (BHV::hamming_distance(query_bhv, result_bhv) == 0) {
                found = true;
                break;
            }
        }
        if (found) {
            recall_at_10_count++;
        }
    }

    double recall_at_10 = static_cast<double>(recall_at_10_count) / NUM_BHVS;
    std::cout << "Recall@10: " << recall_at_10 << std::endl;
    ASSERT_GE(recall_at_10, 0.9);
}
