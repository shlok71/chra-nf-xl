cmake_minimum_required(VERSION 3.18)
project(DNR)

# DNR Module CMakeLists.txt

add_library(dnr STATIC
    src/dnr.cpp
)

target_include_directories(dnr PUBLIC include)


