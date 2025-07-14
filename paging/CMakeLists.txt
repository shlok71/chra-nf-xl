# CMake configuration for the Paging System library

# --- Paging System Library ---
add_library(paging_system STATIC
    src/paging_system.cpp
)
target_include_directories(paging_system PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
set_target_properties(paging_system PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)
target_compile_features(paging_system PUBLIC cxx_std_17)
