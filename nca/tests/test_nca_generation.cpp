#include <gtest/gtest.h>
#include "nca_canvas.h"

TEST(NCAGenerationTest, GenerateImage) {
    NCACanvas canvas(10, 10);
    BHV seed = BHV::encode("seed");
    std::vector<uint8_t> image = canvas.generate_image(seed);
    ASSERT_EQ(image.size(), 100);
}

TEST(NCAGenerationTest, GenerateVideo) {
    NCACanvas canvas(10, 10);
    BHV seed = BHV::encode("seed");
    std::vector<std::vector<uint8_t>> video = canvas.generate_video(seed, 5);
    ASSERT_EQ(video.size(), 5);
    ASSERT_EQ(video[0].size(), 100);
}
