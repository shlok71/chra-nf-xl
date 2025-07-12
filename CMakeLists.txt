cmake_minimum_required(VERSION 3.18)
project(CHRA-NF-XL CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Platform-Specific Setup ---

if(WIN32)
    # Windows (MSVC)
    message(STATUS "Configuring for Windows")
    set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "Vcpkg toolchain file")
    add_definitions(-D_WIN32_WINNT=0x0601)
elseif(APPLE)
    # macOS
    message(STATUS "Configuring for macOS")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3")
else()
    # Linux
    message(STATUS "Configuring for Linux")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -mavx2")
endif()

# --- Subdirectories ---
# These will be populated in later phases.
add_subdirectory(bvh)
add_subdirectory(shm)
add_subdirectory(router)
# add_subdirectory(edgt)
# add_subdirectory(nca)
# add_subdirectory(dnr)
# add_subdirectory(inference)

# --- Enable Testing ---
enable_testing()

# --- Python Bindings (Example using Pybind11) ---
# find_package(pybind11 CONFIG)
# if(pybind11_FOUND)
#     message(STATUS "Pybind11 found, Python bindings can be built.")
# else()
#     message(WARNING "Pybind11 not found. Python bindings will be skipped.")
# endif()

# --- Final Message ---
message(STATUS "CHRA-NF-XL configuration complete.")
