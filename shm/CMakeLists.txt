# CMake configuration for the SHM library

# --- SHM Library ---
add_library(shm STATIC
    src/shm_index.cpp
)
target_include_directories(shm PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(shm PRIVATE bvh)
set_target_properties(shm PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)
target_compile_features(shm PUBLIC cxx_std_17)


