#ifndef PARTICLE_BEHAVIOURS
#define PARTICLE_BEHAVIOURS

#include <cuda.h>
#include <cuda_runtime.h>

#include "Float3Funcs.h"

struct Particle
{
	float3 position;
	float scale;
	float3 color;
};

struct ParticleData
{
	float3 m_velocity;
	float3 m_deltaDirection;
	float3 m_respawnPosition;
	float m_lifetime; // Seconds until this particle is destroyed.
};

struct GlobalData 
{
	float3 target; // A target particles should move towards if used.
	float3 targetVelocity; // The velocity of the target point.
	float3 origin; // Worldspace origin of particle emission.
	float3 startColor; // Color of all particles when the simulation starts.
	float accelerationRate;
	float maxVelocity; // Particles cannot exceed this velocity magnitude.
	float deltaTime;
	float scaleDecay;
	float velocityDecay;
	float spawnScale; // Scale of particle when it is respawned.
	float lifeTime; // Amount of time a particle will live before being killed. (If they can be killed when out of lifetime).
	float respawnTime;
	float burstInterval; // Time interval between bursts.
	int burstCount; // Amount of particles per emission burst.
	int burstRespawnID; // The index of the particle burst to respawn this frame.
	int particleCount; // Amount of particles present in the particlesystem.
	int blockSize; // Thread block size.
};

// ------------------------------------------------------------------------------------------------------------------------------------------
// Color by lifetime functions

typedef void(*ColorFunc)(Particle&, ParticleData&, GlobalData&);

__device__ void Rainbow(Particle& mainData, ParticleData& miscData, GlobalData& globalData)
{
	float3& currentColor = mainData.color;
	float lifeTime = fmodf(fabsf(miscData.m_lifetime), 3.01f);
	
	currentColor = { 0.0f, 0.0f, 0.0f };

	if (lifeTime < 0.33f)
		currentColor.x = 1.0f;
	if (lifeTime > 0.22f && lifeTime < 0.76f)
		currentColor.y = 1.0f;
	if (lifeTime > 0.66f)
		currentColor.z = 1.0f;
}

// ------------------------------------------------------------------------------------------------------------------------------------------
// Velocity-based movement and steering behaviours.

typedef void(*SteerFunc)(Particle&, ParticleData&, GlobalData&);

__device__ void Seek(Particle& mainData, ParticleData& miscData, GlobalData& globalData) 
{
	// Position reference.
	float3& position = mainData.position;

	// Calculate desired velocity.
	float3 targetVelocity = Mul (Normalize(Sub(globalData.target, position)), globalData.accelerationRate);

	// Add to existing velocity.
	miscData.m_velocity = Add(miscData.m_velocity, Mul(Sub(targetVelocity, miscData.m_velocity), globalData.deltaTime));
}

__device__ void Flee(Particle& mainData, ParticleData& miscData, GlobalData& globalData)
{
	// Position reference.
	float3& position = mainData.position;

	// Calculate desired velocity.
	float3 targetVelocity = Mul(Normalize(Sub(position, globalData.target)), globalData.accelerationRate);

	// Add to existing velocity.
	miscData.m_velocity = Add(miscData.m_velocity, Mul(Sub(targetVelocity, miscData.m_velocity), globalData.deltaTime));
}

__device__ void Pursue(Particle& mainData, ParticleData& miscData, GlobalData& globalData)
{
	// Position reference.
	float3& position = mainData.position;

	// Calculate desired velocity.
	float3 velocityAddon = Mul(Normalize(Sub(Add(globalData.target, globalData.targetVelocity), position)), globalData.accelerationRate);

	// Add to existing velocity.
	miscData.m_velocity = Add(miscData.m_velocity, Mul(Sub(velocityAddon, miscData.m_velocity), globalData.deltaTime));
}

__device__ void Evade(Particle& mainData, ParticleData& miscData, GlobalData& globalData)
{
	// Position reference.
	float3& position = mainData.position;

	// Calculate desired velocity.
	float3 velocityAddon = Mul(Normalize(Sub(position, Add(globalData.target, globalData.targetVelocity))), globalData.accelerationRate);

	// Add to existing velocity.
	miscData.m_velocity = Add(miscData.m_velocity, Mul(Sub(velocityAddon, miscData.m_velocity), globalData.deltaTime));
}

__device__ void ClampVelocity(Particle& mainData, ParticleData& miscData, GlobalData& globalData) 
{
	float3& velocity = miscData.m_velocity;
	float mag = Magnitude(velocity);

	if(mag > globalData.maxVelocity) 
	{
		velocity = Mul(Normalize(velocity), globalData.maxVelocity);
	}
}

__device__ void AddGravity(Particle& mainData, ParticleData& miscData, GlobalData& globalData)
{
	miscData.m_velocity = Add(miscData.m_velocity, Mul({ 0.0f, -9.81f, 0.0f }, globalData.deltaTime));
}

// ------------------------------------------------------------------------------------------------------------------------------------------
// Kill functions

typedef void(*KillFunc)(Particle&, ParticleData&, GlobalData&);

__device__ __forceinline__ void Kill(Particle& mainData, ParticleData& miscData, GlobalData& globalData) 
{
	mainData.scale = 0.0f;
	miscData.m_lifetime = 0.0f;
}

// Kill a particle if it enters the specified radius from the target point.
__device__ void KillRadiusTarget(Particle& mainData, ParticleData& miscData, GlobalData& globalData, const float& radius)
{
	float dist = Distance(mainData.position, globalData.target);

	// Kill particle if it is not outside of the kill radius.
	// Particle is considered dead if it's scale is zero.
	if(dist <= radius) 
	{
		Kill(mainData, miscData, globalData);
	}
}

__device__ void KillOutOfRadiusTarget(Particle& mainData, ParticleData& miscData, GlobalData& globalData, const float& radius)
{
	float dist = Distance(mainData.position, globalData.target);

	// Kill particle if it is outside of the kill radius.
	// Particle is considered dead if it's scale is zero.
	if (dist > radius)
	{
		Kill(mainData, miscData, globalData);
	}
}

// ------------------------------------------------------------------------------------------------------------------------------------------
// Respawn functions

typedef void(*RespawnFunc)(Particle&, ParticleData&, GlobalData&, const int&);

__device__ void Respawn(Particle& mainData, ParticleData& miscData, GlobalData& globalData, const int& index) 
{
	mainData.position = Add(globalData.origin, miscData.m_respawnPosition);
	mainData.scale = globalData.spawnScale;
	miscData.m_velocity = { 0.0f, 0.0f, 0.0f };
	miscData.m_lifetime = globalData.lifeTime;
}

__device__ void RespawnBurst(Particle& mainData, ParticleData& miscData, GlobalData& globalData, const int& index)
{
	// The burst ID of this particle is it's index divided by the burst count.
	if (index / globalData.burstCount == globalData.burstRespawnID)
		Respawn(mainData, miscData, globalData, index);
}

__device__ void RespawnIfKilled(Particle& mainData, ParticleData& miscData, GlobalData& globalData, const int& index)
{
	if(mainData.scale == 0.0f && miscData.m_lifetime <= -globalData.respawnTime) 
	{
		Respawn(mainData, miscData, globalData, index);
	}
}

__device__ void DontRespawn(Particle& mainData, ParticleData& miscData, GlobalData& globalData, const int& index) 
{

}

// ------------------------------------------------------------------------------------------------------------------------------------------
// Initial spawn functions.

typedef void(*PosFunction)(Particle&, ParticleData&, GlobalData&, const int&);

__device__ void PosGrid(Particle& particle, ParticleData& particleData, GlobalData& globalData, const int& index)
{
	const float dimensions = 500.0f;

	// Square root of the particle count rounded down.
	int edgeCount = (int)sqrtf((float)globalData.particleCount);

	// Rounded down square count.
	int edgeCountSqr = edgeCount * edgeCount;

	float lenDiv = dimensions / edgeCount;

	int edgeIndex = index % edgeCount;

	float xPos = edgeIndex * lenDiv;
	float zPos = (index / edgeCount) * lenDiv;

	// Reset z of particles outside of bounds.
	if (index >= edgeCountSqr)
		zPos = 0.0f;

	// Set data...
	particleData.m_respawnPosition = { xPos, 0.0f, zPos };
	particle.position = Add(globalData.origin, particleData.m_respawnPosition);
	particle.scale = 0.0f;
	particle.color = globalData.startColor;
}

// ------------------------------------------------------------------------------------------------------------------------------------------

#endif