cmake_minimum_required(VERSION 3.16)
project(pharmaparticlesim LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(include)

add_executable(pharmaparticlesim
    src/main.cpp
    src/simulator.cpp
    src/contact_model.cpp
    src/material_database.cpp
)

target_include_directories(pharmaparticlesim PRIVATE include)