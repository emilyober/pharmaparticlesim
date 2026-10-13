cmake_minimum_required(VERSION 3.16)
project(pharmaparticlesim)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

add_executable(pharmaparticlesim
    src/main.cpp
    src/particle.cpp
    src/simulator.cpp
)