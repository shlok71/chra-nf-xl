# CMake configuration for the System Detection library

# --- System Detection Library ---
add_library(system_detection STATIC
    src/system_detection.cpp
)
target_include_directories(system_detection PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
set_target_properties(system_detection PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)
target_compile_features(system_detection PUBLIC cxx_std_17)
