#include "tts_wrapper.h"
#include <espeak-ng/speak_lib.h>

TTSWrapper::TTSWrapper() {
    espeak_Initialize(AUDIO_OUTPUT_PLAYBACK, 0, NULL, 0);
}

void TTSWrapper::speak(const std::string& text) {
    espeak_Synth(text.c_str(), text.length() + 1, 0, POS_CHARACTER, 0, espeakCHARS_AUTO, NULL, NULL);
    espeak_Synchronize();
}
