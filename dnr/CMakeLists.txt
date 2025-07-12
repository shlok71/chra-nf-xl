cmake_minimum_required(VERSION 3.18)
project(DNR)

# DNR Module CMakeLists.txt

add_library(dnr STATIC
    src/dnr.cpp
)

target_include_directories(dnr PUBLIC include)

add_executable(test_dnr tests/test_dnr.cpp)
target_link_libraries(test_dnr PRIVATE dnr)

# Python Bindings
set(pybind11_DIR /home/jules/.pyenv/versions/3.12.11/lib/python3.12/site-packages/pybind11/share/cmake/pybind11)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(dnr_py python/dnr_py.cpp)
target_link_libraries(dnr_py PRIVATE dnr)
