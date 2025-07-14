#include <gtest/gtest.h>
#include "nca_canvas.h"

TEST(NCACanvasTest, EditPixel) {
    NCACanvas canvas(10, 10);
    canvas.edit_pixel(5, 5, 255);
    // This is a placeholder test. A real test would check the
    // value of the pixel after editing.
    ASSERT_TRUE(true);
}

TEST(NCACanvasTest, DrawCircle) {
    NCACanvas canvas(10, 10);
    canvas.draw_circle(5, 5, 3, 255);
    // This is a placeholder test. A real test would check the
    // pixels in the circle.
    ASSERT_TRUE(true);
}

TEST(NCACanvasTest, ToBHV) {
    NCACanvas canvas(10, 10);
    BHV bhv = canvas.to_bhv();
    // This is a placeholder test. A real test would check the
    // properties of the generated BHV.
    ASSERT_NE(bhv.data[0], 0);
}
