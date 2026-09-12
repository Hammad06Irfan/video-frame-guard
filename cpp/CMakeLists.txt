cmake_minimum_required(VERSION 3.16)
project(AMDFrameGuard CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Discover Homebrew's installed onnxruntime headers and libraries
execute_process(
    COMMAND brew --prefix onnxruntime
    OUTPUT_VARIABLE ONNXRUNTIME_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

include_directories(${ONNXRUNTIME_PREFIX}/include/onnxruntime ${CMAKE_CURRENT_SOURCE_DIR})
link_directories(${ONNXRUNTIME_PREFIX}/lib)

add_executable(frame_guard main.cpp)
target_link_libraries(frame_guard PRIVATE onnxruntime)