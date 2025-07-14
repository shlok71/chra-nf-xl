# CMake configuration for the Quantizer library

# --- Quantizer Library ---
add_library(quantizer STATIC
    src/quantizer.cpp
)
target_include_directories(quantizer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
set_target_properties(quantizer PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)
target_compile_features(quantizer PUBLIC cxx_std_17)
