#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

inline void checkCuda(cudaError_t result, const char* operation){
    if (result != cudaSuccess){
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}