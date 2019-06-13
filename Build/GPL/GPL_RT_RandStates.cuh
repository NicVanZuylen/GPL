#ifndef INIT_RAND
#define INIT_RAND

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "device_launch_parameters.h"

__device__ curandState_t* randStates;

__global__ void InitRandKernel(curandState_t* states, unsigned int seed) 
{
	// Get thread index.
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	// Initialize curand states.
	curand_init(seed + i, 0, 0, &states[i]);
}

extern "C" void InitRandLaunch(void* d_states, unsigned int seed, unsigned int count, unsigned int blockSize) 
{
	// Copy device pointer to device location.
	cudaMemcpyToSymbol(randStates, d_states, sizeof(void*));

	InitRandKernel<<<count / blockSize, blockSize >>>((curandState_t*)d_states, seed);
}

#endif