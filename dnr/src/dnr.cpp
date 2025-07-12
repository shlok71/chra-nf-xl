#include "dnr.h"
#include <iostream>
#include <cmath>

DNR::DNR(int width, int height) : width(width), height(height), image(width * height * 4, 0) {}

void DNR::rasterize(const std::vector<Point>& controlPoints) {
    // This is a placeholder for the spline rasterizer
    // It should take control points and produce a raster image
    // For now, we'll just draw lines between the control points
    if (controlPoints.size() < 2) {
        return;
    }

    for (size_t i = 0; i < controlPoints.size() - 1; ++i) {
        const Point& p1 = controlPoints[i];
        const Point& p2 = controlPoints[i+1];

        int x1 = static_cast<int>(p1.x * width);
        int y1 = static_cast<int>(p1.y * height);
        int x2 = static_cast<int>(p2.x * width);
        int y2 = static_cast<int>(p2.y * height);

        // Bresenham's line algorithm
        int dx = std::abs(x2 - x1);
        int dy = std::abs(y2 - y1);
        int sx = (x1 < x2) ? 1 : -1;
        int sy = (y1 < y2) ? 1 : -1;
        int err = dx - dy;

        while (true) {
            if (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height) {
                int index = (y1 * width + x1) * 4;
                image[index] = 255;     // R
                image[index + 1] = 255; // G
                image[index + 2] = 255; // B
                image[index + 3] = 255; // A
            }

            if (x1 == x2 && y1 == y2) {
                break;
            }

            int e2 = 2 * err;
            if (e2 > -dy) {
                err -= dy;
                x1 += sx;
            }
            if (e2 < dx) {
                err += dx;
                y1 += sy;
            }
        }
    }
}

const std::vector<uint8_t>& DNR::getImage() const {
    return image;
}
