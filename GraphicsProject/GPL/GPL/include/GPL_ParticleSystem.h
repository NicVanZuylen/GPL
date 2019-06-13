#pragma once
#include "GPL.h"
#include "GPL_VertexAttributes.h"
#include "GPL_Behaviour.h"

struct CUstream_st;
struct cudaGraphicsResource;

namespace GPL 
{
	struct float3 
	{
		float x;
		float y;
		float z;
	};

	typedef CUstream_st* cudaStream_t;

	// Vertex structure for particles used for drawing.
	struct GPL_Particle3D
	{
		float3 m_position; // Worldspace position.
		float m_scale; // Scale multiplier.
		float3 m_color; // Draw color tint.
	};

	// Template structure for local particle data.
	struct GPL_Particle3DData
	{
		float3 m_velocity;
		float3 m_deltaDirection;
		float3 m_respawnPosition;
		float m_lifetime; // Seconds until this particle is destroyed.
	};

	// Structure containing data that is likely useful in all particle systems.
	struct GPL_BaseData 
	{
		int m_particleCount;
		int m_blockSize;
		int m_burstCount;
		int m_burstRespawnID;
		int m_currentBurstID;
		float m_burstInterval;
		float m_deltaTime;
	};

	// Template structure for global Particlesystem data.
	struct GPL_GlobalP3DData
	{
		float3 m_target; // A target particles should move towards if used.
		float3 m_targetVelocity; // The velocity of the target point.
		float3 m_origin; // Worldspace origin of particle emission.
		float3 m_startColor; // Color of all particles when the simulation starts.
		float m_accelerationRate; // Rate particles can accelerate using steering behaviours.
		float m_maxVelocity; // Particles cannot exceed this velocity magnitude.
		float m_scaleDecay;
		float m_velocityDecay;
		float m_spawnScale; // Scale of the particle when it is spawned.
		float m_lifeTime; // Amount of time a particle will live before being killed. (If they can be killed when out of lifetime).
		float m_respawnTime;
	};

	// Rand initialization function.
	extern "C" void InitRandLaunch(void* d_states, unsigned int seed, unsigned int count, unsigned int blockSize);

	class GPL_ParticleSystem 
	{
	public:

		typedef void* devicePtr;

		// Default constructor
		GPL_ParticleSystem();

		// Overloaded constructor for custom particle count and particle kernel.
		GPL_ParticleSystem(GPL_BehaviourProgram* program, unsigned int particleCount);

		// Overloaded constructor for templated data structures.
		// T: Global data structure shared across all particles.
		// U: Data structure for each particle instance.
		template<typename T, typename U>
		GPL_ParticleSystem(GPL_BehaviourProgram* program, unsigned int particleCount) 
		{
			m_d_randStates = nullptr;

			m_baseData.m_particleCount = particleCount;
			m_baseData.m_burstCount = 100;
			m_baseData.m_burstInterval = 0.5f;
			m_baseData.m_burstRespawnID = 0;
			m_baseData.m_currentBurstID = 0;
			m_baseData.m_blockSize = GPL_Main::BlockSize();

			m_globalSize = sizeof(T);
			m_dataSize = sizeof(U);

			m_behaviourProgram = program;

			GenBuffers();
		}

		// Destructor
		~GPL_ParticleSystem();

		/*
		Description: Set initial particle data.
		Param:
		    unsigned int randomSeed: The initial seed for the gpu-side random number generator.
		*/
		void Initialize(unsigned int randomSeed = 12345);

		template<typename T>
		void SetGlobalData(T* data) 
		{
			std::memcpy(m_globalData, data, sizeof(T));
		}

		/*
		Description: Create a CUDA stream for this Particlesystem.
		*/
		static void CreateStream();

		/*
		Description: Destroy the CUDA stream created for this ParticleSystem.
		*/
		static void DestroySteam();

		/*
		Description: Sync the CUDA stream with the CPU.
		*/
		static void Sync();

		/*
		Description: Get the CUDA stream used for this ParticleSystem. (If it exists.)
		Return Type: cudaStream_t&
		*/
		static cudaStream_t& GetStream();

		/*
		Description: Set basic data that will be used in the particle system. NOTE: Some data will not overwrite existing data.
		Param:
		    GPL_BaseData& data: The data to use.
		*/
		void SetBaseData(GPL_BaseData& data);

		/*
		Description: Set the mesh particles will use to render.
		Param:
		    unsigned int glVBOHandle: The OpenGL handle to the mesh vertex buffer.
			unsigned int glIndexBufferHandle: The OpenGL handle to the mesh index buffer.
			GPL_VertexAttributes& attributes: The vertex attribute structure for the vertex buffer.
		*/
		void SetMesh(unsigned int glVBOHandle, unsigned int glIndexBufferHandle, unsigned int indexCount, GPL_VertexAttributes& attributes);

		/*
		Description: Change the amount of particles in the particlesystem.
		Param:
		    unsigned int particleCount: The new amount of particles.
		*/
		void Resize(unsigned int particleCount);

		/*
		Description: Draw the particles using the provided mesh handle.
		Param:
		    unsigned int glVAOHandle: The VAO handle to use. Should contain a mesh vertex buffer and an index buffer of unsigned integers.
			unsigned int indexCount: The amount of indices in the index buffer.
		*/
		void Draw();

		/*
		Description: Draw all particles as points instead of meshes.
		*/
		void DrawSimple();

		/*
		Description: Launches GPU kernels that will update all particles in this particle system.
		Param:
		     const float& deltaTime: Deltatime used in CUDA kernels for particle updates.
		*/
		void Update(const float& deltaTime);

		// Getters and setters:

		const unsigned int& GetVAOHandle();

	private:

		// Generate particle buffers...
		void GenBuffers();

		// Update without a CUDA stream.
		inline void UpdateDefaultStream(float deltaTime);

		// PRNGs
		devicePtr m_d_randStates;
		devicePtr m_d_randFunc;

		// Particles
		//GPL_GlobalP3DData m_globalData;
		GPL_BaseData m_baseData;
		unsigned char* m_globalData;
		float m_burstTimer; // Host side timer to increment burst ID every time the burst interval is reached.

		// Templated data sizes...
		unsigned int m_globalSize;
		unsigned int m_dataSize;

		// GL handles
		unsigned int m_glParticleVAO;
		unsigned int m_glParticleVBO; // Instanced particle VBO.
		unsigned int m_glParticleData; // Data used by the particle kernel, it is stored in a OpenGL buffer for fast access through GL interop.

		// Mesh
		unsigned int m_indexCount;

		// CUDA handles
		GPL_BehaviourProgram* m_behaviourProgram;

		static cudaStream_t m_stream;
		static bool m_streamExists;

		devicePtr m_drawPtr;
		devicePtr m_dataPtr;

		cudaGraphicsResource* m_drawResource;
		cudaGraphicsResource* m_dataResource;
	};
};