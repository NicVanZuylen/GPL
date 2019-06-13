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

	localData.m_respawnPosition = globalData.origin + (offsetDir * 5.0f);
	localParticle.position = localData.m_respawnPosition;

	float velMag = 5.0f + (RAND_UNIFORM(&randStates[i]) * 5.0f * (5.0f + zDir));

	localData.m_velocity = vec3(0.0f, 0.0f, 1.0f) * velMag;
}

extern "C" __global__ void UpdateKernel(GLOBAL_TYPE globalData, void* drawBuffer, void* dataBuffer, BaseData baseData, xorwow_state32_t* randStates)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	Particle* particles = (Particle*)drawBuffer;
	ParticleData* data = (ParticleData*)dataBuffer;

	Particle& localParticle = particles[i];
	ParticleData& localData = data[i];

	if (ShouldBurst(baseData, i)) 
	{
		float xDir = RAND_UNIFORM(&randStates[i]);
		float yDir = RAND_UNIFORM(&randStates[i]);
		float zDir = RAND_UNIFORM(&randStates[i]);

		xDir = (xDir * 2.0f) - 1.0f;
		yDir = (yDir * 2.0f) - 1.0f;
		zDir = (zDir * 2.0f) - 1.0f;

		vec3 offsetDir = vec3(xDir, yDir, zDir).Normalized();

		localData.m_respawnPosition = globalData.origin + (offsetDir * 5.0f);
		localParticle.position = localData.m_respawnPosition;

		float velMag = 5.0f + (RAND_UNIFORM(&randStates[i]) * 5.0f * (1.0f + zDir));

		localData.m_velocity = vec3(0.0f, 0.0f, 1.0f) * velMag;

		localParticle.scale = globalData.spawnScale;
	}

	if(localParticle.scale > 0.0f)
	{
		localParticle.position = localParticle.position + (localData.m_velocity * baseData.m_deltaTime);
	}
}

