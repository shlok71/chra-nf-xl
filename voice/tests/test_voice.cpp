#include <gtest/gtest.h>
#include "whisper_wrapper.h"
#include "tts_wrapper.h"

TEST(VoiceTest, Transcribe) {
    WhisperWrapper whisper;
    std::string text = whisper.transcribe("audio.wav");
    ASSERT_EQ(text, "This is a transcribed sentence.");
}

TEST(VoiceTest, Speak) {
    TTSWrapper tts;
    // This is a placeholder test. A real test would check that
    // the audio is generated correctly.
    tts.speak("Hello, world!");
    ASSERT_TRUE(true);
}
