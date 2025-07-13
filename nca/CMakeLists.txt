cmake_minimum_required(VERSION 3.18)
project(NCA)

# NCA Module CMakeLists.txt

add_library(nca STATIC
    src/nca.cpp
)

target_include_directories(nca PUBLIC include ../third_party/MiniDNN-master/include)

set(pybind11_DIR /home/jules/.pyenv/versions/3.12.11/lib/python3.12/site-packages/pybind11/share/cmake/pybind11)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(nca_py python/nca_py.cpp)
target_link_libraries(nca_py PRIVATE nca)


