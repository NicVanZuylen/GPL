#ifndef INIT_RAND
#define INIT_RAND

#include <cuda.h>
#include <cuda_runtime.h>
#include "GPL/thirdparty/rngpu.hpp"

#include "device_launch_parameters.h"

__device__ float Rand(xorwow_state32_t* states, int index)
{
	return uniform_float_xorwow32(&states[index]);
}

__global__ void InitRandKernel(xorwow_state32_t* states, unsigned int seed)
{
	// Get thread index.
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	// Initialize curand states.
	states[i] = get_initial_xorwow_state32(seed + i);
}

extern "C"  void InitRandLaunch(void* d_states, unsigned int seed, unsigned int count, unsigned int blockSize) 
{
	int blockCount = (count / blockSize) + 1;

	if (count < blockSize) 
		blockSize = count;

	// Launch initialization kernel.
	InitRandKernel<<<blockCount, blockSize>>>((xorwow_state32_t*)d_states, seed);
}

#endif