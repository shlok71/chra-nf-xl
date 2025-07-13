#ifndef NCA_H
#define NCA_H

#include <vector>
#include <cstdint>
#include "../../third_party/MiniDNN-master/include/MiniDNN.h"

#include <deque>

#include "../../bvh/include/bhv.h"

class NCA {
public:
    NCA(int width, int height);
    void step();
    const std::vector<uint8_t>& getGrid() const;
    void load_weights(const std::string& filename);
    void set_pixel(int x, int y, uint8_t r, uint8_t g, uint8_t b, uint8_t a);
    void draw_circle(int cx, int cy, int radius, uint8_t r, uint8_t g, uint8_t b, uint8_t a);
    void undo();
    void redo();
    void settle();
    BHV transcribe_to_bhv();

private:
    int width;
    int height;
    std::vector<uint8_t> grid;
    MiniDNN::Network mlp;
    std::deque<std::vector<uint8_t>> history;
    int history_index;
    void save_history();
};

#endif // NCA_H
