# CMake configuration for the Router library

# --- Find ONNX Runtime ---
# This assumes ONNX Runtime is installed in a standard location.
# For CI, we'll need to handle its installation.
# find_package(onnxruntime REQUIRED)

# --- Router Library ---
add_library(router STATIC
    src/pmr.cpp
)
target_include_directories(router PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(router PRIVATE bvh /home/jules/.pyenv/versions/3.12.11/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.22.1)
set_target_properties(router PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)
target_compile_features(router PUBLIC cxx_std_17)



# --- Copy ONNX model to build directory ---
add_custom_command(
    TARGET router_tests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/model/router.onnx
            $<TARGET_FILE_DIR:router_tests>/router.onnx
)
add_custom_command(
    TARGET router_py POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/model/router.onnx
            $<TARGET_FILE_DIR:router_py>/router.onnx
)
