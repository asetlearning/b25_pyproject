cmake_minimum_required(VERSION 3.20)
project(b25_pyproject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# Suppress tcgraph's own tests/examples from being built by default
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)

add_subdirectory(cpp/tcgraph EXCLUDE_FROM_ALL)  # provides tc_graph target
add_subdirectory(cpp)
