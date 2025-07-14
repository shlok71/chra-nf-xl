# CMake configuration for the Profiler library

# --- Profiler Library ---
add_library(profiler STATIC
    src/profiler.cpp
)
target_include_directories(profiler PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(profiler PRIVATE psutil)
set_target_properties(profiler PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)
target_compile_features(profiler PUBLIC cxx_std_17)
