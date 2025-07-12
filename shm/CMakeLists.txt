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

# --- GoogleTest ---
find_package(GTest REQUIRED)
add_executable(shm_tests
    tests/test_shm_index.cpp
)
target_link_libraries(shm_tests PRIVATE shm GTest::gtest_main)
add_test(NAME shm_tests COMMAND shm_tests)

# --- Python Bindings (Pybind11) ---
set(pybind11_DIR /home/jules/.pyenv/versions/3.12.11/lib/python3.12/site-packages/pybind11/share/cmake/pybind11)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(shm_py
    python/shm_py.cpp
)
target_link_libraries(shm_py PRIVATE shm)
