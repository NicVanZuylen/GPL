#pragma once

#include "GPL_RT_ParticleBehaviours.cuh"
#include "rngpu.hpp"

#ifndef __global__
#define __global__
#define __host__
#define __device__
#define __forceinline__

#endif

#define GLOBAL_TYPE GlobalData

#define RAND_UNIFORM uniform_float_xorwow32

extern "C" __global__ void InitKernel(GLOBAL_TYPE globalData, void* drawBuffer, void* dataBuffer, BaseData baseData, xorwow_state32_t* randStates)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	Particle* particles = (Particle*)drawBuffer;
	ParticleData* data = (ParticleData*)dataBuffer;

	Particle& localParticle = particles[i];
	ParticleData& localData = data[i];

	localParticle.scale = 0.0f;
	localParticle.color = globalData.startColor;

	float xDir = RAND_UNIFORM(&randStates[i]);
	float yDir = RAND_UNIFORM(&randStates[i]);
	float zDir = RAND_UNIFORM(&randStates[i]);

	xDir = (xDir * 2.0f) - 1.0f;
	yDir = (yDir * 2.0f) - 1.0f;
	zDir = (zDir * 2.0f) - 1.0f;

	vec3 offsetDir = vec3(xDir, yDir, zDir).Normalized();

	localData.m_respawnPosition = globalData.origin + (offsetDir * 50.0f);
	localParticle.position = localData.m_respawnPosition;
}

extern "C" __global__ void UpdateKernel(GLOBAL_TYPE globalData, void* drawBuffer, void* dataBuffer, BaseData baseData, xorwow_state32_t* randStates)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	Particle* particles = (Particle*)drawBuffer;
	ParticleData* data = (ParticleData*)dataBuffer;

	Particle& localParticle = particles[i];
	ParticleData& localData = data[i];

	if(localParticle.scale > 0.0f)
	{
		Seek(localParticle, localData.m_velocity, 10.0f, baseData, globalData.target);
        
        localParticle.position += localData.m_velocity * baseData.m_deltaTime;

		if(localParticle.position.Distance(globalData.target) <= 1.0f)
			localParticle.scale = 0.0f;
	}
	else if(ShouldBurst(baseData, i))
	{
        localParticle.position = localData.m_respawnPosition;
        localParticle.scale = 0.01f;
	}
}

