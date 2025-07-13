#include <gtest/gtest.h>
#include "bhv.h"

TEST(BHVTest, EncodeDecode) {
    std::string input = "hello world";
    BHV encoded = BHV::encode(input);
    std::string decoded = BHV::decode(encoded);
    ASSERT_EQ(input, decoded);
}

TEST(BHVTest, Bind) {
    BHV a = BHV::encode("hello");
    BHV b = BHV::encode("world");
    BHV c = BHV::bind(a, b);
    // This is a placeholder test. A real test would check the
    // properties of the bound vector.
    ASSERT_NE(c.data[0], 0);
}

TEST(BHVTest, Bundle) {
    std::vector<BHV> bhvs;
    bhvs.push_back(BHV::encode("hello"));
    bhvs.push_back(BHV::encode("world"));
    BHV bundled = BHV::bundle(bhvs);
    // This is a placeholder test. A real test would check the
    // properties of the bundled vector.
    ASSERT_NE(bundled.data[0], 0);
}
