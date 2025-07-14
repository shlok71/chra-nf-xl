# CMake configuration for the Overlay library

# --- Overlay Library ---
add_library(overlay STATIC
    src/overlay.cpp
)
target_include_directories(overlay PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
set_target_properties(overlay PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)
target_compile_features(overlay PUBLIC cxx_std_17)
