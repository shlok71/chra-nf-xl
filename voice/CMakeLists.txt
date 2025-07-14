# CMake configuration for the Voice library

# --- Voice Library ---
add_library(voice STATIC
    input/whisper_wrapper.cpp
    output/tts_wrapper.cpp
)
target_include_directories(voice PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/input
    ${CMAKE_CURRENT_SOURCE_DIR}/output
)
set_target_properties(voice PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)
target_compile_features(voice PUBLIC cxx_std_17)
