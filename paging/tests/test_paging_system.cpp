#include <gtest/gtest.h>
#include "paging_system.h"

TEST(PagingSystemTest, PageOutIn) {
    PagingSystem paging_system;
    std::string module_name = "test_module";
    std::vector<uint8_t> data = {1, 2, 3, 4, 5};
    paging_system.page_out(module_name, data);
    std::vector<uint8_t> paged_in_data = paging_system.page_in(module_name);
    ASSERT_EQ(data, paged_in_data);
}
