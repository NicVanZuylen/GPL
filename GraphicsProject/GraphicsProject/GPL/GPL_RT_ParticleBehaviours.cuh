#ifndef PARTICLE_BEHAVIOURS
#define PARTICLE_BEHAVIOURS

#include "GPL_RT_Vectors.cuh"

struct Particle
{
	vec3 position;
	float scale;
	vec3 color;
};

struct ParticleData
{
	vec3 m_velocity;
	vec3 m_deltaDirection;
	vec3 m_respawnPosition;
	float m_lifetime; // Seconds until this particle is destroyed.
};

struct BaseData 
{
	int m_particleCount;
	int m_blockSize;
	int m_burstCount;
	int m_burstRespawnID;
	int m_currentBurstID;
	float m_burstInterval;
	float m_deltaTime;
};

struct GlobalData 
{
	vec3 target; // A target particles should move towards if used.
	vec3 targetVelocity; // The velocity of the target point.
	vec3 origin; // Worldspace origin of particle emission.
	vec3 startColor; // Color of all particles when the simulation starts.
	float accelerationRate;
	float maxVelocity; // Particles cannot exceed this velocity magnitude.
	float scaleDecay;
	float velocityDecay;
	float spawnScale; // Scale of particle when it is respawned.
	float lifeTime; // Amount of time a particle will live before being killed. (If they can be killed when out of lifetime).
	float respawnTime;
};

// ------------------------------------------------------------------------------------------------------------------------------------------
// Velocity-based movement and steering behaviours.

__device__ __forceinline__ void Seek(Particle& mainData, vec3& velocity, const float& accelerationRate, BaseData& globalData, const vec3& target)
{
	// Position reference.
	vec3& position = mainData.position;

	// Calculate desired velocity.
    vec3 targetVelocity = Normalize(target - position) * accelerationRate;

	// Add to existing velocity.
	velocity = velocity + ((targetVelocity - velocity) * globalData.m_deltaTime);
}

__device__ __forceinline__ void Flee(Particle& mainData, vec3& velocity, const float& accelerationRate, BaseData& globalData, const vec3& target)
{
	// Position reference.
	vec3& position = mainData.position;

	// Calculate desired velocity.

    vec3 targetVelocity = Normalize(position - target) * accelerationRate;

	// Add to existing velocity.
	velocity = velocity + ((targetVelocity - velocity) * globalData.m_deltaTime);
}

__device__ __forceinline__ void Pursue(Particle& mainData, vec3& velocity, const vec3& targetVelocity, const float& accelerationRate, BaseData& globalData, const vec3& target)
{
	// Position reference.
	vec3& position = mainData.position;

	// Calculate desired velocity.
    vec3 velocityAddon = Normalize((target + targetVelocity) - position) * accelerationRate;

	// Add to existing velocity.
	velocity = velocity + ((velocityAddon - velocity) * globalData.m_deltaTime);
}

__device__ __forceinline__ void Evade(Particle& mainData, vec3& velocity, const vec3& targetVelocity, const float& accelerationRate, BaseData& globalData, const vec3& target)
{
	// Position reference.
	vec3& position = mainData.position;

	// Calculate desired velocity.
    vec3 velocityAddon = Normalize(position - (target + targetVelocity)) * accelerationRate;

	// Add to existing velocity.
	velocity = velocity + ((velocityAddon - velocity) * globalData.m_deltaTime);
}

__device__ __forceinline__ void ClampVelocity(vec3& velocity, const float& maxVelocity)
{
	float mag = Magnitude(velocity);

	if(mag > maxVelocity) 
	{
		velocity = velocity.Normalized() * maxVelocity;
	}
}

__device__ __forceinline__ void AddGravity(Particle& mainData, vec3& velocity, BaseData& globalData)
{
	velocity = velocity + (vec3(0.0f, -9.81f, 0.0f) * globalData.m_deltaTime);
}

__device__ __forceinline__ bool ShouldBurst(BaseData& baseData, const int& index)
{
	// The burst ID of this particle is it's index divided by the burst count.
	return index / baseData.m_burstCount == baseData.m_burstRespawnID;
}

#endif
