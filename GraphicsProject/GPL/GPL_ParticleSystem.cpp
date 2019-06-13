#include "GPL_ParticleSystem.h"
#include "GLAD/glad.h"
#include <iostream>
#include <cmath>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <vector_types.h>
#include <cuda_gl_interop.h>
#include "GPL/thirdparty/rngpu.hpp"

#include <fstream>
#include <sstream>

using namespace GPL;

cudaStream_t GPL_ParticleSystem::m_stream = (CUstream_st*)0;
bool GPL_ParticleSystem::m_streamExists = false;

GPL_ParticleSystem::GPL_ParticleSystem() 
{
	m_d_randStates = nullptr;

	m_baseData.m_particleCount = 100;
	m_baseData.m_burstCount = 100;
	m_baseData.m_burstInterval = 0.5f;
	m_baseData.m_burstRespawnID = 0;
    m_baseData.m_currentBurstID = 0;
	m_baseData.m_blockSize = GPL_Main::BlockSize();

	m_globalSize = sizeof(GPL_GlobalP3DData);
	m_dataSize = sizeof(GPL_Particle3DData);

	m_behaviourProgram = nullptr;

	GenBuffers();
}

GPL_ParticleSystem::GPL_ParticleSystem(GPL_BehaviourProgram* program, unsigned int particleCount) 
{
	m_d_randStates = nullptr;

	m_baseData.m_particleCount = particleCount;
	m_baseData.m_burstCount = 100;
	m_baseData.m_burstInterval = 0.5f;
	m_baseData.m_burstRespawnID = 0;
	m_baseData.m_currentBurstID = 0;
	m_baseData.m_blockSize = GPL_Main::BlockSize();

	m_globalSize = sizeof(GPL_GlobalP3DData);
	m_dataSize = sizeof(GPL_Particle3DData);

	m_behaviourProgram = program;

	GenBuffers();
}

GPL_ParticleSystem::~GPL_ParticleSystem() 
{
	// Free random memory.
	cudaFree(m_d_randStates);

	// Unregister buffers from CUDA.
	cudaGraphicsUnregisterResource(m_drawResource);
	cudaGraphicsUnregisterResource(m_dataResource);

	// Delete global data store.
	delete[] m_globalData;

	// Delete buffers.
	glDeleteVertexArrays(1, &m_glParticleVAO);
	glDeleteBuffers(1, &m_glParticleVBO);
	glDeleteBuffers(1, &m_glParticleData);
}

void GPL_ParticleSystem::Initialize(unsigned int randomSeed)
{
	// Create stream.
	cudaStream_t stream;
	cudaStreamCreate(&stream);

	// Map buffers.
	cudaGraphicsMapResources(1, &m_drawResource, stream);
	cudaGraphicsMapResources(1, &m_dataResource, stream);

	// Get mapped device pointers.
	size_t drawSize;
	size_t dataSize;
	cudaGraphicsResourceGetMappedPointer(&m_drawPtr, &drawSize, m_drawResource);
	cudaGraphicsResourceGetMappedPointer(&m_dataPtr, &dataSize, m_dataResource);

	// Allocate memory for random states and values...
	cudaMalloc(&m_d_randStates, sizeof(xorwow_state32_t) * m_baseData.m_particleCount);

	// Launch PRNG initialization kernel.
	InitRandLaunch(m_d_randStates, randomSeed, m_baseData.m_particleCount, m_baseData.m_blockSize);

	// Function parameters for JIT initialization kernel.
	void* params[] =
	{
		m_globalData,
		&m_drawPtr,
		&m_dataPtr,
		&m_baseData,
		&m_d_randStates
	};

	// Function handle.
	CUfunction function = m_behaviourProgram->InitFunction();

	// Thread data.
	int blockSize = m_baseData.m_blockSize;
	int blockCount = (m_baseData.m_particleCount / blockSize) + 1;

	if (m_baseData.m_particleCount < blockSize)
		blockSize = m_baseData.m_particleCount;

	// Launch...
	CUresult result = cuLaunchKernel(function, blockCount, 1, 1, blockSize, 1, 1, 0, stream, params, 0);

	// Destroy stream.
	cudaStreamDestroy(stream);
}

void GPL_ParticleSystem::CreateStream() 
{
	if(!m_streamExists) 
	{
		cudaStreamCreate(&m_stream);
		m_streamExists = true;
	}
}

void GPL_ParticleSystem::DestroySteam() 
{
	if(m_streamExists) 
	{
		cudaStreamDestroy(m_stream);
		m_streamExists = false;
	}
}

void GPL_ParticleSystem::Sync() 
{
	if (m_streamExists)
		cudaStreamSynchronize(m_stream);
}

cudaStream_t& GPL_ParticleSystem::GetStream()
{
	return m_stream;
}

void GPL_ParticleSystem::SetBaseData(GPL_BaseData& data) 
{
	int particleCount = m_baseData.m_particleCount;
	int blockSize = m_baseData.m_blockSize;

	m_baseData = data;
	m_baseData.m_particleCount = particleCount;
	m_baseData.m_blockSize = blockSize;
	m_baseData.m_currentBurstID = 0;
}

void GPL_ParticleSystem::SetMesh(unsigned int glVBOHandle, unsigned int glIndexBufferHandle, unsigned int indexCount, GPL_VertexAttributes& attributes) 
{
	m_indexCount = indexCount;

	// Re-create VAO.
	glDeleteVertexArrays(1, &m_glParticleVAO);
	glGenVertexArrays(1, &m_glParticleVAO);
	glBindVertexArray(m_glParticleVAO);

	// Buffers should already be filled.

	// Vertex buffer
	glBindBuffer(GL_ARRAY_BUFFER, glVBOHandle);
	attributes.UseAttributes(); // Assign attributes...

	// Index buffer
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, glIndexBufferHandle);

	// Instance buffer.
	glBindBuffer(GL_ARRAY_BUFFER, m_glParticleVBO);

	// Vertex attributes.

	unsigned int vertAttribCount = attributes.AttributeCount();

	// Positions...
	glEnableVertexAttribArray(vertAttribCount);
	glVertexAttribPointer(vertAttribCount, 3, GL_FLOAT, GL_FALSE, sizeof(GPL_Particle3D), (void*)0);

	// Scales...
	glEnableVertexAttribArray(vertAttribCount + 1);
	glVertexAttribPointer(vertAttribCount + 1, 1, GL_FLOAT, GL_FALSE, sizeof(GPL_Particle3D), (void*)(sizeof(float) * 3));

	// Colors...
	glEnableVertexAttribArray(vertAttribCount + 2);
	glVertexAttribPointer(vertAttribCount + 2, 3, GL_FLOAT, GL_FALSE, sizeof(GPL_Particle3D), (void*)(sizeof(float) * 4));

	// Divisors.
	glVertexAttribDivisor(vertAttribCount, 1);
	glVertexAttribDivisor(vertAttribCount + 1, 1);
	glVertexAttribDivisor(vertAttribCount + 2, 1);

	// Unbind.
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void GPL_ParticleSystem::Resize(unsigned int particleCount) 
{
	// Unmap buffers.
	cudaGraphicsUnmapResources(1, &m_drawResource, m_stream);
	cudaGraphicsUnmapResources(1, &m_dataResource, m_stream);

	// Unregister buffers from CUDA.
	cudaGraphicsUnregisterResource(m_drawResource);
	cudaGraphicsUnregisterResource(m_dataResource);

	// Create new set.
	m_baseData.m_particleCount = particleCount;

	// Resize buffer.
	glBindBuffer(GL_ARRAY_BUFFER, m_glParticleVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(GPL_Particle3D) * m_baseData.m_particleCount, 0, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, m_glParticleData);
	glBufferData(GL_ARRAY_BUFFER, m_dataSize * m_baseData.m_particleCount, 0, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// Register new buffers with CUDA.
	cudaGraphicsGLRegisterBuffer(&m_drawResource, m_glParticleVBO, cudaGraphicsRegisterFlagsNone);
	cudaGraphicsGLRegisterBuffer(&m_dataResource, m_glParticleData, cudaGraphicsRegisterFlagsNone);

	cudaGraphicsMapResources(1, &m_drawResource, m_stream);
	cudaGraphicsMapResources(1, &m_dataResource, m_stream);

}

void GPL_ParticleSystem::Draw() 
{
	glBindVertexArray(m_glParticleVAO); // VAO should have a mesh VBO and index buffer.
	glBindBuffer(GL_ARRAY_BUFFER, m_glParticleVBO);

	// Draw...
	glDrawElementsInstanced(GL_TRIANGLES, m_indexCount, GL_UNSIGNED_INT, 0, m_baseData.m_particleCount);
}

void GPL_ParticleSystem::DrawSimple() 
{
	glBindVertexArray(m_glParticleVAO);

	glDrawArraysInstanced(GL_POINTS, 0, 1, m_baseData.m_particleCount);
}

void GPL_ParticleSystem::Update(const float& deltaTime) 
{
	if(!m_streamExists) 
	{
		UpdateDefaultStream(deltaTime);
		return;
	}

	// Set deltatime.
	m_baseData.m_deltaTime = deltaTime;

	// Update bursts...
	m_burstTimer += deltaTime;

	// No bursts will occur with id: -1
	// If the current burst ID is not incremented this frame the value will remain -1.
	m_baseData.m_burstRespawnID = -1;

	if(m_burstTimer >= m_baseData.m_burstInterval)
	{
		m_burstTimer = 0.0f;

		// Increment and keep burst ID within range.
		++m_baseData.m_currentBurstID %= (static_cast<int>(ceil(static_cast<float>(m_baseData.m_particleCount) / static_cast<float>(m_baseData.m_burstCount))));
	   
		// Apply burst ID for this frame.
		m_baseData.m_burstRespawnID = m_baseData.m_currentBurstID;
	}

	// Function parameters for JIT initialization kernel.
	void* params[] =
	{
		m_globalData,
		&m_drawPtr,
		&m_dataPtr,
		&m_baseData,
		&m_d_randStates
	};

	// Function handle.
	const CUfunction& function = m_behaviourProgram->UpdateFunction();

	// Thread data.
	int blockSize = m_baseData.m_blockSize;
	int blockCount = (m_baseData.m_particleCount / blockSize) + 1;

	if (m_baseData.m_particleCount < blockSize)
		blockSize = m_baseData.m_particleCount;

	// Launch...
	CUresult result = cuLaunchKernel(function, blockCount, 1, 1, blockSize, 1, 1, 0, m_stream, params, 0);

	// Unmap buffer.
}

void GPL_ParticleSystem::UpdateDefaultStream(float deltaTime) 
{
	cudaError_t error = cudaGraphicsMapResources(1, &m_drawResource);

	error = cudaGraphicsMapResources(1, &m_dataResource);

	// Get mapped pointers.
	void* drawPtr = nullptr;
	void* dataPtr = nullptr;
	error = cudaGraphicsResourceGetMappedPointer(&drawPtr, (size_t*)0, m_drawResource);
	error = cudaGraphicsResourceGetMappedPointer(&dataPtr, (size_t*)0, m_dataResource);

	// Set deltatime.
	m_baseData.m_deltaTime = deltaTime;

	// Update bursts...
	m_burstTimer += deltaTime;

	// No bursts will occur with id: -1
	// If the current burst ID is not incremented this frame the value will remain -1.
	m_baseData.m_burstRespawnID = -1;

	if (m_burstTimer >= m_baseData.m_burstInterval)
	{
		m_burstTimer = 0.0f;

		// Increment and keep burst ID within range.
		++m_baseData.m_currentBurstID %= ((int)ceil((float)m_baseData.m_particleCount / (float)m_baseData.m_burstCount));

		// Apply burst ID for this frame.
		m_baseData.m_burstRespawnID = m_baseData.m_currentBurstID;
	}

	// Function parameters for JIT initialization kernel.
	void* params[] =
	{
		m_globalData,
		&drawPtr,
		&dataPtr,
		&m_baseData,
		&m_d_randStates
	};

	// Function handle.
	CUfunction function = m_behaviourProgram->UpdateFunction();

	// Thread data.
	int blockSize = m_baseData.m_blockSize;
	int blockCount = (m_baseData.m_particleCount / blockSize) + 1;

	if (m_baseData.m_particleCount < blockSize)
		blockSize = m_baseData.m_particleCount;

	// Launch...
	CUresult result = cuLaunchKernel(function, blockCount, 1, 1, blockSize, 1, 1, 0, 0, params, 0);

	// Unmap buffer.
	drawPtr = nullptr;
	dataPtr = nullptr;
	error = cudaGraphicsUnmapResources(1, &m_drawResource);
	error = cudaGraphicsUnmapResources(1, &m_dataResource);
}

void GPL_ParticleSystem::GenBuffers() 
{
	m_glParticleVBO = 0;
	m_glParticleData = 0;

	// Particle instance VBO
	glGenVertexArrays(1, &m_glParticleVAO);
	glBindVertexArray(m_glParticleVAO);

	glGenBuffers(1, &m_glParticleVBO);
	glBindBuffer(GL_ARRAY_BUFFER, m_glParticleVBO);

	// Buffer data store.
	glBufferData(GL_ARRAY_BUFFER, sizeof(GPL_Particle3D) * m_baseData.m_particleCount, 0, GL_DYNAMIC_DRAW);

	// Vertex attributes.

	// Positions...
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(GPL_Particle3D), (void*)0);

	// Scales...
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(GPL_Particle3D), (void*)(sizeof(float) * 3)); 

	// Colors...
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(GPL_Particle3D), (void*)(sizeof(float) * 4));

	// Divisors.
	glVertexAttribDivisor(0, 1);
    glVertexAttribDivisor(1, 1);
	glVertexAttribDivisor(2, 1);

	// Unbind VAO.
	glBindVertexArray(0);

	// Particle data

	// Generate and bind buffer.
	glGenBuffers(1, &m_glParticleData);
	glBindBuffer(GL_ARRAY_BUFFER, m_glParticleData);

	// Fill buffer.
	// No VAO and vertex attributes are required, as this buffer will not be used for drawing.
	glBufferData(GL_ARRAY_BUFFER, m_dataSize * m_baseData.m_particleCount, 0, GL_DYNAMIC_DRAW);

	// Unbind.
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// Register buffers with CUDA.
	cudaGraphicsGLRegisterBuffer(&m_drawResource, m_glParticleVBO, cudaGraphicsRegisterFlagsNone);
	cudaGraphicsGLRegisterBuffer(&m_dataResource, m_glParticleData, cudaGraphicsRegisterFlagsNone);

	// Allocate global data...
	m_globalData = new unsigned char[m_globalSize];
}

const unsigned int& GPL_ParticleSystem::GetVAOHandle() 
{
	return m_glParticleVAO;
}