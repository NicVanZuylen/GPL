#pragma once
#include "Mesh.h"
#include "Vector4.h"
#include "Matrix4.h"

class Material;

class StaticMeshRenderer 
{
public:

	StaticMeshRenderer(Material* material);

	~StaticMeshRenderer();

	/*
	Description: Draw all static meshes contained within this static mesh buffer.
	*/
	void Draw();

	/*
	Description: Add a mesh to the static mesh buffer, transformed with the provided model matrix.
	Param:
	    Mesh* mesh: The mesh to copy into the static mesh buffer.
		const float* modelMatrixData: The model matrix to transform the mesh in float ptr format.
	*/
	void PushMesh(Mesh* mesh, const float* modelMatrixData);

	/*
	Description: Pushes the current static mesh buffer to the GPU, ready to be drawn.
	*/
	void FinalizeBuffers();

	/*
	Description: Set the material used to draw this static mesh.
	Param:
	    Material* material: The material used for drawing this static mesh. 
	*/
	void SetMaterial(Material* material);

private:

	unsigned char* ResizeBuffer(unsigned char* buffer, unsigned int nSize, unsigned int nOldSize);

	struct Instance
	{
		NVZMathLib::Vector4 m_v4Color;
		float m_modelMat[16];
		float m_normalMat[9];
	};

	Material* m_material;

	unsigned char* m_staticVertices;
	unsigned char* m_staticIndices;

	unsigned int m_nStaticVertSize;
	unsigned int m_nUsedVertSpace;
	unsigned int m_nStaticIndSize;
	unsigned int m_nUsedIndSpace;

	unsigned int m_glStaticVAOHandle;
	unsigned int m_glStaticVBOHandle;
	unsigned int m_glStaticEBOHandle;
	unsigned int m_glStaticInstanceHandle;

	unsigned int m_nMaterialIndex;
};
