#include <gtest/gtest.h>
#include "sgr.h"

TEST(SGRTest, Route) {
    SGR sgr;
    BHV text_bhv = BHV::encode("text");
    BHV ocr_bhv = BHV::encode("ocr");
    BHV canvas_bhv = BHV::encode("canvas");
    BHV retrieval_bhv = BHV::encode("retrieval");

    ASSERT_EQ(sgr.route(text_bhv), TaskType::TEXT);
    ASSERT_EQ(sgr.route(ocr_bhv), TaskType::OCR);
    ASSERT_EQ(sgr.route(canvas_bhv), TaskType::CANVAS);
    ASSERT_EQ(sgr.route(retrieval_bhv), TaskType::RETRIEVAL);
}

TEST(SGRTest, LazyLoad) {
    SGR sgr;
    bool loaded = false;
    sgr.register_module(TaskType::TEXT, [&]() {
        loaded = true;
    });

    sgr.route(BHV::encode("text"));
    ASSERT_TRUE(loaded);
}
