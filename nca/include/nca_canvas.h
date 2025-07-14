#pragma once

#include "bhv.h"
#include <vector>
#include <string>

class NCACanvas {
public:
    NCACanvas(int width, int height);
    void edit_pixel(int x, int y, uint8_t value);
    void draw_circle(int cx, int cy, int r, uint8_t value);
    void step();
    BHV to_bhv();
    std::vector<uint8_t> generate_image(const BHV& seed);
    std::vector<std::vector<uint8_t>> generate_video(const BHV& seed, int num_frames);

private:
    int width;
    int height;
    std::vector<uint8_t> grid;
};
