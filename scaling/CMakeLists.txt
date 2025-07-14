# CMake configuration for the Scaling Engine library

# --- Scaling Engine Library ---
add_library(scaling_engine STATIC
    src/scaling_engine.cpp
)
target_include_directories(scaling_engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(scaling_engine PRIVATE system_detection)
set_target_properties(scaling_engine PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)
target_compile_features(scaling_engine PUBLIC cxx_std_17)
