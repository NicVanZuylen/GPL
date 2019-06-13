#pragma once
#include "Vector4.h"
#include "Matrix4.h"
#include "glm.hpp"

class Mesh;
class Material;
class Shader;

#define MAX_INSTANCE_COUNT 512

#define MAX_MAP_COUNT 16

class Batch
{
public:

	Batch(Mesh* mesh, Material* material);

	~Batch();

	/*
	Description: Add an instance to be rendered this frame.
	Param:
	    float* modelMatrix: The model matrix of the instance. 
		Vector4 v4Color: The color of the instance.
	*/
	void Add(float* modelMatrix, NVZMathLib::Vector4 v4Color = { 1.0f, 1.0f, 1.0f, 1.0f });

	/*
	Description: Send the current state of the batch to the GPU and draw. And make way for the more instances.
	*/
	void Flush();

private:

	void SetData();

	void CreateBuffers();

	struct Vertex 
	{
		NVZMathLib::Vector4 m_v4Position;
		NVZMathLib::Vector4 m_v4TexCoords;
	};

	struct Instance 
	{
		NVZMathLib::Vector4 m_v4Color;
		float m_modelMat[16];
		float m_normalMat[9];
	};

	//unsigned int m_glVAOHandle;
	//unsigned int m_glVBOHandle;
	//unsigned int m_glInsHandle;
	//unsigned int m_glEBOHandle;

	Vertex m_vertices[8];
	unsigned int m_indices[36];

	Instance m_instances[MAX_INSTANCE_COUNT];
	unsigned int m_nInstanceCount;

	unsigned int m_nSamplerLocations[MAX_MAP_COUNT];

	Mesh* m_mesh;
	Material* m_material;
};

