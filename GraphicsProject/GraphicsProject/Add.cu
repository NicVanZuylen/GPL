// CUDA kernel file of addition test.

#include <cuda.h>

__global__ void Add(int count, int* a, int* b) 
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < count; i += stride)
		a[i] = i * 2;
}

extern "C" void RunKernel(int count, int* aVals, int* bVals) 
{
	const int blockSize = 256;
	int blockCount = (count + blockSize - 1) / blockSize;

    Add<<<blockCount, blockSize>>>(count, aVals, bVals);
}