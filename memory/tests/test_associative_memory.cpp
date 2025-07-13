#include <gtest/gtest.h>
#include "associative_memory.h"

TEST(AssociativeMemoryTest, InsertAndQuery) {
    AssociativeMemory memory;
    BHV key = BHV::encode("key");
    BHV value = BHV::encode("value");
    memory.insert(key, value);
    BHV result = memory.query(key);
    ASSERT_EQ(BHV::hamming_distance(result, value), 0);
}

TEST(AssociativeMemoryTest, Update) {
    AssociativeMemory memory;
    BHV key = BHV::encode("key");
    BHV value1 = BHV::encode("value1");
    BHV value2 = BHV::encode("value2");
    memory.insert(key, value1);
    memory.update(key, value2);
    BHV result = memory.query(key);
    BHV expected = BHV::bundle({value1, value2});
    ASSERT_EQ(BHV::hamming_distance(result, expected), 0);
}
