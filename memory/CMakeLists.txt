# CMake configuration for the Associative Memory library

# --- Associative Memory Library ---
add_library(memory STATIC
    src/associative_memory.cpp
)
target_include_directories(memory PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(memory PRIVATE bvh)
set_target_properties(memory PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)
target_compile_features(memory PUBLIC cxx_std_17)
