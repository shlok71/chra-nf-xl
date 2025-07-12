#include "dnr.h"
#include <cassert>
#include <iostream>
#include <vector>

void test_dnr_initialization() {
    DNR dnr(128, 128);
    const auto& image = dnr.getImage();
    assert(image.size() == 128 * 128 * 4);
    std::cout << "test_dnr_initialization passed" << std::endl;
}

void test_dnr_rasterize() {
    DNR dnr(128, 128);
    std::vector<Point> controlPoints = {{0.1, 0.1}, {0.9, 0.9}};
    dnr.rasterize(controlPoints);
    const auto& image = dnr.getImage();
    // Check if some pixels have been written
    bool pixel_written = false;
    for (size_t i = 0; i < image.size(); i += 4) {
        if (image[i] != 0 || image[i+1] != 0 || image[i+2] != 0 || image[i+3] != 0) {
            pixel_written = true;
            break;
        }
    }
    assert(pixel_written);
    std::cout << "test_dnr_rasterize passed" << std::endl;
}

int main() {
    test_dnr_initialization();
    test_dnr_rasterize();
    return 0;
}
