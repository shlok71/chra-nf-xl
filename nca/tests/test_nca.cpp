#include "nca.h"
#include <cassert>
#include <iostream>

void test_nca_initialization() {
    NCA nca(256, 256);
    const auto& grid = nca.getGrid();
    assert(grid.size() == 256 * 256);
    std::cout << "test_nca_initialization passed" << std::endl;
}

void test_nca_step() {
    NCA nca(256, 256);
    auto grid_before = nca.getGrid();
    nca.step();
    const auto& grid_after = nca.getGrid();
    assert(grid_before != grid_after);
    std::cout << "test_nca_step passed" << std::endl;
}

int main() {
    test_nca_initialization();
    test_nca_step();
    return 0;
}
