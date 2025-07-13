# CMake configuration for the BVH library

# --- Compiler Feature Detection for AVX2 ---
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-mavx2" COMPILER_SUPPORTS_AVX2)

if(COMPILER_SUPPORTS_AVX2)
    message(STATUS "AVX2 support detected. Enabling -mavx2 flag.")
    set(AVX2_FLAGS "-mavx2")
elseif(MSVC)
    message(STATUS "MSVC compiler detected. Assuming AVX2 support is available via intrinsics.")
    set(AVX2_FLAGS "") # MSVC enables intrinsics by default
else()
    message(WARNING "AVX2 support not detected. Performance may be suboptimal.")
    set(AVX2_FLAGS "")
endif()

# --- BVH Library ---
add_library(bvh STATIC
    src/avx2_bhv.cpp
)
target_include_directories(bvh PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
set_target_properties(bvh PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)
target_compile_features(bvh PUBLIC cxx_std_17)
target_compile_options(bvh PRIVATE ${AVX2_FLAGS})


