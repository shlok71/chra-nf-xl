# CMake configuration for the Compressor library

# --- Compressor Library ---
add_library(compressor STATIC
    src/compressor.cpp
)
target_include_directories(compressor PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
set_target_properties(compressor PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)
target_compile_features(compressor PUBLIC cxx_std_17)
