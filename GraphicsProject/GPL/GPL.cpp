#include "GPL.h"
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

using namespace GPL;

CUdevice GPL_Main::m_device = 0;
cudaDeviceProp* GPL_Main::m_properties = nullptr;
CUcontext GPL_Main::m_context = 0;

int GPL_Main::Init()
{
	// Get graphics device and set the cuda device to it.
	auto error = cudaGetDevice(&m_device);

	if (error)
	{
		std::cout << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
		return GPL_INVALID_DEVICE;
	}

	// Get device information.
	m_properties = new cudaDeviceProp;

	auto propError = cudaGetDeviceProperties(m_properties, m_device);

	if (propError)
	{
		std::cout << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
		return GPL_INVALID_DEVICE_PROPERTIES;
	}

	// Set CUDA device.
	auto setError = cudaSetDevice(m_device);

	if (setError)
	{
		std::cout << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
		return GPL_COULD_NOT_SET_DEVICE;
	}

	// Create context for legacy CUDA functions necessary for execution of runtime compiled kernels.
	cuInit(0);
	cuCtxCreate(&m_context, 0, m_device);

	return 0;
}

void GPL_Main::Quit() 
{
	delete m_properties;

	cuCtxDestroy(m_context);
}

int GPL_Main::BlockSize() 
{
	return m_properties->maxThreadsDim[0] / 2;
}

int GPL_Main::GridSize() 
{
	return m_properties->maxGridSize[0];
}