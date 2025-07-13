#ifndef DNR_H
#define DNR_H

#include <vector>
#include <cstdint>

struct Point {
    float x, y;
};

class DNR {
public:
    DNR(int width, int height);
    void rasterize(const std::vector<Point>& controlPoints);
    const std::vector<uint8_t>& getImage() const;

private:
    int width;
    int height;
    std::vector<uint8_t> image;
};

#endif // DNR_H
