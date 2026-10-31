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

option(ENABLE_CUDA "Build CUDA verification targets" OFF)
if(ENABLE_CUDA)
    if(CMAKE_VERSION VERSION_LESS "3.18")
        message(FATAL_ERROR "CUDA targets require CMake 3.18 or newer")
    endif()
    enable_language(CUDA)
    add_executable(cuda_integration_check
        verification/cuda/check_integration.cu
        src/cuda/device_particles.cu
        src/cuda/integrate.cu
        src/cuda/reset_forces.cu
    )
    target_include_directories(cuda_integration_check PRIVATE include)
    set_target_properties(cuda_integration_check PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_EXTENSIONS OFF
    )
    enable_testing()
    add_test(NAME cuda_integration COMMAND cuda_integration_check)
    add_executable(cuda_pair_contact_check
    verification/cuda/check_pair_contact.cu
    src/cuda/device_particles.cu
    src/cuda/reset_forces.cu
    src/cuda/pair_contact.cu
    src/contact_model.cpp
    src/material_database.cpp
    )
    target_include_directories(cuda_pair_contact_check PRIVATE include)

    set_target_properties(cuda_pair_contact_check PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_EXTENSIONS OFF
    )

    add_test(
        NAME cuda_pair_contact
        COMMAND cuda_pair_contact_check "${CMAKE_CURRENT_SOURCE_DIR}"
    )
endif()