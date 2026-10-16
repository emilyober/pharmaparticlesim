cmake_minimum_required(VERSION 3.16)

project(dem_viewer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(raylib CONFIG REQUIRED)

add_executable(dem_viewer viewer.cpp)

target_link_libraries(dem_viewer PRIVATE raylib)