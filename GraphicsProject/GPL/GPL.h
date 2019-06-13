#pragma once

// Errors
#define GPL_INVALID_DEVICE 1
#define GPL_INVALID_DEVICE_PROPERTIES 2
#define GPL_COULD_NOT_SET_DEVICE 3

struct CUctx_st;
struct cudaDeviceProp;

namespace GPL 
{
	typedef void* devicePtr;

	typedef int cudaDevice;
	typedef CUctx_st* cudaContext;

	class GPL_Main 
	{
	public:

		// Description: Initialize GPL. This will also initialize a CUDA context.
		static int Init();

		// Description: Quit use of GPL and free it's resources.
		static void Quit();

		// Size of the thread block used for executing particle kernels.
		static int BlockSize();

		// Size of the thread block grid used for executing particle kernels.
		static int GridSize();

	private:

		static cudaDevice m_device;
		static cudaDeviceProp* m_properties;
		static cudaContext m_context;
	};
};
