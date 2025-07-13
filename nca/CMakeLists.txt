cmake_minimum_required(VERSION 3.18)
project(NCA)

# NCA Module CMakeLists.txt

add_library(nca STATIC
    src/nca.cpp
)

target_include_directories(nca PUBLIC include)


