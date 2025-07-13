#ifndef NCA_H
#define NCA_H

#include <vector>
#include <cstdint>

class NCA {
public:
    NCA(int width, int height);
    void step();
    const std::vector<uint8_t>& getGrid() const;

private:
    int width;
    int height;
    std::vector<uint8_t> grid;
    // 5-layer MLP would be a separate class, but for now, we'll keep it simple
    void applyMLP(int x, int y);
};

#endif // NCA_H
