# CMake configuration for the Reasoning Engine library

# --- Reasoning Engine Library ---
add_library(reasoning STATIC
    src/reasoning_engine.cpp
)
target_include_directories(reasoning PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(reasoning PRIVATE bvh)
set_target_properties(reasoning PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)
target_compile_features(reasoning PUBLIC cxx_std_17)
