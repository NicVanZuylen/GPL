#include "StaticMeshRenderer.h"
#include "Matrix3.h"
#include "Material.h"
#include "Shader.h"
#include "GLAD\glad.h"
#include "glm.hpp"

StaticMeshRenderer::StaticMeshRenderer(Material* material) 
{
	SetMaterial(material);

	m_material->Use();

	for (int i = 0; i < m_material->MapCount(); ++i)
	{
		// Get the name of the current sampler block...
		std::string szSamplerName = "textureMaps[" + std::to_string(i) + "]";

		unsigned int glSamplerLocation = glGetUniformLocation(m_material->GetShader()->GetHandle(), szSamplerName.c_str());

		// Assign sampler location...
		glUniform1i(glSamplerLocation, i);
	}

	m_staticVertices = nullptr;
	m_staticIndices = nullptr;

	m_nStaticVertSize = 0;
	m_nStaticIndSize = 0;
	m_nUsedVertSpace = 0;
	m_nUsedIndSpace = 0;

	m_glStaticVAOHandle = 0;
	m_glStaticVBOHandle = 0;
	m_glStaticEBOHandle = 0;
	m_glStaticInstanceHandle = 0;

	// Generate and bind VAO.
	glGenVertexArrays(1, &m_glStaticVAOHandle);
	glBindVertexArray(m_glStaticVAOHandle);

	// Bind static index buffer to this VAO.
	glGenBuffers(1, &m_glStaticEBOHandle);

	// Bind static vertex buffer to this VAO.
	glGenBuffers(1, &m_glStaticVBOHandle);
	glBindBuffer(GL_ARRAY_BUFFER, m_glStaticVBOHandle);

	// Vertex attributes...

	// Positions
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(float) * 14, (void*)0);
	glEnableVertexAttribArray(0);

	// Normals
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(float) * 14, (void*)(sizeof(float) * 4));
	glEnableVertexAttribArray(1);

	// Tangents
	glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(float) * 14, (void*)(sizeof(float) * 8));
	glEnableVertexAttribArray(2);

	// Texture Coordinates
	glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 14, (void*)(sizeof(float) * 12));
	glEnableVertexAttribArray(3);

	// Create and bind instance buffer.
	glGenBuffers(1, &m_glStaticInstanceHandle);
	glBindBuffer(GL_ARRAY_BUFFER, m_glStaticInstanceHandle);

	// Create dummy instance with dummy matrices.
	Instance instance;
	memcpy_s(instance.m_modelMat, sizeof(float) * 16, &NVZMathLib::Matrix4(), sizeof(float) * 16);
	memcpy_s(instance.m_normalMat, sizeof(float) * 9, &NVZMathLib::Matrix3(), sizeof(float) * 9);
	instance.m_v4Color = { 1.0f, 1.0f, 1.0f, 1.0f };

	// Fill with single instance data.
	glBufferData(GL_ARRAY_BUFFER, sizeof(Instance) * 1, &instance, GL_STATIC_DRAW);

	// Instance attributes...

	// Color
	glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, sizeof(Instance), (void*)0);
	glEnableVertexAttribArray(4);

	glVertexAttribDivisor(4, 1);

	// Model matrix
	glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, sizeof(Instance), (void*)(sizeof(float) * 4));
	glEnableVertexAttribArray(5);

	glVertexAttribPointer(6, 4, GL_FLOAT, GL_FALSE, sizeof(Instance), (void*)(sizeof(float) * 8));
	glEnableVertexAttribArray(6);

	glVertexAttribPointer(7, 4, GL_FLOAT, GL_FALSE, sizeof(Instance), (void*)(sizeof(float) * 12));
	glEnableVertexAttribArray(7);

	glVertexAttribPointer(8, 4, GL_FLOAT, GL_FALSE, sizeof(Instance), (void*)(sizeof(float) * 16));
	glEnableVertexAttribArray(8);

	glVertexAttribDivisor(5, 1);
	glVertexAttribDivisor(6, 1);
	glVertexAttribDivisor(7, 1);
	glVertexAttribDivisor(8, 1);

	// Normal matrix
	glVertexAttribPointer(9, 3, GL_FLOAT, GL_FALSE, sizeof(Instance), (void*)(sizeof(float) * 20));
	glEnableVertexAttribArray(9);

	glVertexAttribPointer(10, 3, GL_FLOAT, GL_FALSE, sizeof(Instance), (void*)(sizeof(float) * 23));
	glEnableVertexAttribArray(10);

	glVertexAttribPointer(11, 3, GL_FLOAT, GL_FALSE, sizeof(Instance), (void*)(sizeof(float) * 26));
	glEnableVertexAttribArray(11);

	glVertexAttribDivisor(9, 1);
	glVertexAttribDivisor(10, 1);
	glVertexAttribDivisor(11, 1);

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

StaticMeshRenderer::~StaticMeshRenderer() 
{
	if(m_staticVertices)
	    delete[] m_staticVertices;

	if(m_staticIndices)
	    delete[] m_staticIndices;

	glDeleteVertexArrays(1, &m_glStaticVAOHandle);
	glDeleteBuffers(1, &m_glStaticVBOHandle);
	glDeleteBuffers(1, &m_glStaticEBOHandle);
	glDeleteBuffers(1, &m_glStaticInstanceHandle);
}

void StaticMeshRenderer::Draw() 
{
	glBindVertexArray(m_glStaticVAOHandle);
	glBindBuffer(GL_ARRAY_BUFFER, m_glStaticVBOHandle);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_glStaticEBOHandle);
	glBindBuffer(GL_ARRAY_BUFFER, m_glStaticInstanceHandle);

	glDrawElements(GL_TRIANGLES, m_nUsedIndSpace / sizeof(unsigned int), GL_UNSIGNED_INT, 0);

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void StaticMeshRenderer::PushMesh(Mesh* mesh, const float* modelMatrixData) 
{
	NVZMathLib::Matrix4 modelMatrix;
	memcpy_s(&modelMatrix, sizeof(float) * 16, modelMatrixData, sizeof(float) * 16);

	NVZMathLib::Matrix3 normalMat
	(
		modelMatrixData[0], modelMatrixData[1], modelMatrixData[2],
		modelMatrixData[4], modelMatrixData[5], modelMatrixData[6],
		modelMatrixData[8], modelMatrixData[9], modelMatrixData[10]
	);

	unsigned int glMeshVBO = mesh->VBOHandle();
	unsigned int glMeshEBO = mesh->IndexBufferHandle();

	unsigned int nMeshSize = mesh->VertexCount() * sizeof(Mesh::Vertex);
	unsigned int nIndexSize = mesh->IndexCount() * sizeof(unsigned int);

	// -----------------------------------------------------------------------------------
	// Vertices

	glBindBuffer(GL_ARRAY_BUFFER, glMeshVBO);
	unsigned char* meshData = (unsigned char*)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);

	int nPrevMeshVertCount = m_nUsedVertSpace / sizeof(Mesh::Vertex);

	if(m_staticVertices == nullptr) 
	{
		m_staticVertices = new unsigned char[nMeshSize];

		memcpy_s(m_staticVertices, nMeshSize, meshData, nMeshSize);

		// Cast vertex array to usable type for indexing and data access.
		Mesh::Vertex* castVertices = (Mesh::Vertex*)m_staticVertices;

		// Transform all vertices by the model matrix.
		for (unsigned int i = m_nUsedVertSpace / sizeof(Mesh::Vertex); i < mesh->VertexCount(); ++i)
		{
			castVertices[i].m_v4Position = modelMatrix * castVertices[i].m_v4Position;

			// Transform normal and tangent.
			NVZMathLib::Vector3* normalPtr = (NVZMathLib::Vector3*)&castVertices[i].m_v4Normal;
			NVZMathLib::Vector3* tangentPtr = (NVZMathLib::Vector3*)&castVertices[i].m_v4Tangent;

			*normalPtr = normalMat * *normalPtr;
			*tangentPtr = normalMat * *tangentPtr;
		}

		m_nStaticVertSize = nMeshSize;
		m_nUsedVertSpace = nMeshSize;
	}
	else 
	{
		if (m_nStaticVertSize < m_nUsedVertSpace + nMeshSize)
		{
			m_staticVertices = ResizeBuffer(m_staticVertices, m_nUsedVertSpace + nMeshSize, m_nStaticVertSize);

			memcpy_s(&m_staticVertices[m_nUsedVertSpace], nMeshSize, meshData, nMeshSize);

			// Update buffer size and usage data.
			m_nStaticVertSize = m_nUsedVertSpace + nMeshSize;

			// Cast vertex array to usable type for indexing and data access.
			Mesh::Vertex* castVertices = (Mesh::Vertex*)m_staticVertices;

			// Transform all vertices by the model matrix.
			for (unsigned int i = m_nUsedVertSpace / sizeof(Mesh::Vertex); i < m_nStaticVertSize / sizeof(Mesh::Vertex); ++i)
			{
				castVertices[i].m_v4Position = modelMatrix * castVertices[i].m_v4Position;

				// Transform normal and tangent.
				NVZMathLib::Vector3* normalPtr = (NVZMathLib::Vector3*)&castVertices[i].m_v4Normal;
				NVZMathLib::Vector3* tangentPtr = (NVZMathLib::Vector3*)&castVertices[i].m_v4Tangent;

				*normalPtr = normalMat * *normalPtr;
				*tangentPtr = normalMat * *tangentPtr;
			}

			m_nUsedVertSpace = m_nStaticVertSize;
		}
		else
		{
			memcpy_s(&m_staticVertices[m_nUsedVertSpace], nMeshSize, meshData, nMeshSize);

			// Cast vertex array to usable type for indexing and data access.
			Mesh::Vertex* castVertices = (Mesh::Vertex*)m_staticVertices;

			// Transform all vertices by the model matrix.
			for (unsigned int i = m_nUsedVertSpace / sizeof(Mesh::Vertex); i < m_nStaticVertSize / sizeof(Mesh::Vertex); ++i)
			{
				castVertices[i].m_v4Position = modelMatrix * castVertices[i].m_v4Position;

				// Transform normal and tangent.
				NVZMathLib::Vector3* normalPtr = (NVZMathLib::Vector3*)&castVertices[i].m_v4Normal;
				NVZMathLib::Vector3* tangentPtr = (NVZMathLib::Vector3*)&castVertices[i].m_v4Tangent;

				*normalPtr = normalMat * *normalPtr;
				*tangentPtr = normalMat * *tangentPtr;
			}

			// Update buffer usage data.
			m_nUsedVertSpace += nMeshSize;
		}
	}

	glUnmapBuffer(GL_ARRAY_BUFFER);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// -----------------------------------------------------------------------------------
	// Indices

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, glMeshEBO);
	unsigned char* indexData = (unsigned char*)glMapBuffer(GL_ELEMENT_ARRAY_BUFFER, GL_READ_ONLY);

	if (m_staticIndices == nullptr)
	{
		m_staticIndices = new unsigned char[nIndexSize];

		memcpy_s(m_staticIndices, nIndexSize, indexData, nIndexSize);

		m_nStaticIndSize = nIndexSize;
		m_nUsedIndSpace = nIndexSize;
	}
	else 
	{
		if (m_nStaticIndSize < m_nUsedIndSpace + nIndexSize)
		{
			m_staticIndices = ResizeBuffer(m_staticIndices, m_nUsedIndSpace + nIndexSize, m_nStaticIndSize);

			memcpy_s(&m_staticIndices[m_nUsedIndSpace], nIndexSize, indexData, nIndexSize);

			// Update buffer size and usage data.
			m_nStaticIndSize = m_nUsedIndSpace + nIndexSize;

			unsigned int* castIndices = (unsigned int*)m_staticIndices;
			
			unsigned int nUsedIndices = m_nUsedIndSpace / sizeof(unsigned int);
			
			// Add used index count to the new indices, to ensure they do not conflict with the previous mesh.
			for (unsigned int i = nUsedIndices; i < m_nStaticIndSize / sizeof(unsigned int); ++i)
			{
				castIndices[i] += nPrevMeshVertCount;
			}

			m_nUsedIndSpace = m_nStaticIndSize;
		}
		else
		{
			memcpy_s(&m_staticIndices[m_nUsedIndSpace], nIndexSize, indexData, nIndexSize);

			unsigned int* castIndices = (unsigned int*)m_staticIndices;

			unsigned int nUsedIndices = m_nUsedIndSpace / sizeof(unsigned int);

			// Add used index count to the new indices, to ensure they do not conflict with the previous mesh.
			for (int i = nUsedIndices; i < m_nStaticIndSize / sizeof(unsigned int); ++i)
			{
				castIndices[i] += nPrevMeshVertCount;
			}

			// Update buffer usage data.
			m_nUsedIndSpace += nMeshSize;
		}
	}

	glUnmapBuffer(GL_ELEMENT_ARRAY_BUFFER);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	// -----------------------------------------------------------------------------------
}

void StaticMeshRenderer::FinalizeBuffers() 
{
	// Bind buffers to be filled...
	glBindBuffer(GL_ARRAY_BUFFER, m_glStaticVBOHandle);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_glStaticEBOHandle);

	// Fill buffers...
	glBufferData(GL_ARRAY_BUFFER, m_nUsedVertSpace, m_staticVertices, GL_STATIC_DRAW);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_nUsedIndSpace, m_staticIndices, GL_STATIC_DRAW);
}

void StaticMeshRenderer::SetMaterial(Material* material) 
{
	// Remove from old material...
	if(m_material)
	    m_material->GetMeshes().PopAt(m_nMaterialIndex);

	m_material = material;

	// Add to new material if it is not null.
	if (m_material)
	{
		m_nMaterialIndex = m_material->GetStaticMeshes().Count();
		m_material->GetStaticMeshes().Push(this);
	}
	else
		m_nMaterialIndex = 0;
}

unsigned char* StaticMeshRenderer::ResizeBuffer(unsigned char* buffer, unsigned int nSize, unsigned int nOldSize) 
{
	// Allocate new buffer.
	unsigned char* newBuffer = new unsigned char[nSize];

	// Copy old buffer contents into the new one.
	if (nSize > nOldSize)
	{
		memcpy_s(newBuffer, nSize, buffer, nOldSize);
	}
	else
		memcpy_s(newBuffer, nSize, buffer, nSize);

	// Delete old buffer.
	delete[] buffer;

	return newBuffer;
}