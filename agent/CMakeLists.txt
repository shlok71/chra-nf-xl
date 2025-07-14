# CMake configuration for the Agent Framework library

# --- Agent Framework Library ---
add_library(agent_framework STATIC
    src/agent_framework.cpp
)
target_include_directories(agent_framework PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(agent_framework PRIVATE bvh)
set_target_properties(agent_framework PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)
target_compile_features(agent_framework PUBLIC cxx_std_17)
