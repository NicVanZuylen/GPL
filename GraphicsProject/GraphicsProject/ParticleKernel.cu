#pragma once

#include <cmath>

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include "device_launch_parameters.h"

#include <curand_kernel.h>

//#include "RandStates.h"
#include "ParticleBehaviours.h"

__global__ void InitKernel(GlobalData globalData, void* drawBuffer, void* dataBuffer)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	// Arrays
	Particle* particles = (Particle*)drawBuffer;
	ParticleData* data = (ParticleData*)dataBuffer;

	// Particle to modify
	Particle& localParticle = particles[i];
	ParticleData& localData = data[i];

	PosGrid(localParticle, localData, globalData, i);
}

__global__ void ParticleKernel(GlobalData globalData, void* drawBuffer, void* dataBuffer)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	// Arrays
	Particle* particles = (Particle*)drawBuffer;
	ParticleData* data = (ParticleData*)dataBuffer;

	// Particle to modify
	Particle& localParticle = particles[i];
	ParticleData& localData = data[i];



	// Do stuff

	// Respawns this particle if the global burst ID matches that of this particle
	RespawnBurst(localParticle, localData, globalData, i);
	//RespawnIfKilled(localParticle, localData, globalData, i);

	if(localParticle.scale > 0.0f) 
	{
		localData.m_lifetime -= globalData.deltaTime;

		//Pursue(localParticle, localData, globalData);
		//ClampVelocity(localParticle, localData, globalData);
		//KillRadiusTarget(localParticle, localData, globalData, 1.0f);
		Rainbow(localParticle, localData, globalData);
	}

	// Apply velocity to position.
	localParticle.position = Add(localParticle.position, Mul(localData.m_velocity, globalData.deltaTime));
}

extern "C" void InitParticles(GlobalData& data, void* drawBuffer, void* dataBuffer)
{
	int& count = data.particleCount;
	int& blockSize = data.blockSize;

	if (count < blockSize)
		blockSize = count;

	// Launch initialization kernel...
	InitKernel<<<count / blockSize, blockSize>>>(data, drawBuffer, dataBuffer);
}

extern "C" void UpdateParticles(GlobalData& data, void* drawBuffer, void* dataBuffer)
{
	int& count = data.particleCount;
	int& blockSize = data.blockSize;

	if (count < blockSize)
		blockSize = count;

	// Launch update kernel...
	ParticleKernel<<<count / blockSize, blockSize>>>(data, drawBuffer, dataBuffer);
}

