#include "Batch.h"
#include "glad/glad.h"
#include "Material.h"
#include "Shader.h"
#include "Mesh.h"
#include <iostream>
#include "glm/include/ext.hpp"

using namespace NVZMathLib;

Batch::Batch(Mesh* mesh, Material* material)
{
	m_mesh = mesh;
	m_material = material;

	// Use material shader...
	m_material->GetShader()->Use();

	for(int i = 0; i < m_material->MapCount(); ++i) 
	{
		// Get the name of the current sampler block...
		std::string szSamplerName = "textureMaps[" + std::to_string(i) + "]";

		m_nSamplerLocations[i] = glGetUniformLocation(m_material->GetShader()->GetHandle(), szSamplerName.c_str());

		// Assign sampler location...
		glUniform1i(m_nSamplerLocations[i], i);
	}

	// Unbind shader.
	glUseProgram(0);

	SetData();

	CreateBuffers();
}

Batch::~Batch()
{
	m_material = nullptr;

	//glDeleteVertexArrays(1, &m_glVAOHandle);
	//glDeleteBuffers(1, &m_glVBOHandle);
	//glDeleteBuffers(1, &m_glInsHandle);
	//glDeleteBuffers(1, &m_glEBOHandle);
}

void Batch::Add(float* modelMatrix, NVZMathLib::Vector4 v4Color) 
{
	if (m_nInstanceCount >= MAX_INSTANCE_COUNT)
		Flush();

	Instance& currentInstance = m_instances[m_nInstanceCount++];

	// Copy color.
	currentInstance.m_v4Color = v4Color;

	// Copy model matrix.
	memcpy_s(currentInstance.m_modelMat, sizeof(float) * 16, modelMatrix, sizeof(float) * 16);

	// Construct normal matrix.
	glm::mat3 normalMat =
	{
		glm::vec3(modelMatrix[0], modelMatrix[1], modelMatrix[2]),
		glm::vec3(modelMatrix[4], modelMatrix[5], modelMatrix[6]),
		glm::vec3(modelMatrix[8], modelMatrix[9], modelMatrix[10])
	};

	//normalMat = glm::inverseTranspose(normalMat);

	// Copy normal matrix into instance...
	memcpy_s(currentInstance.m_normalMat, sizeof(float) * 9, &normalMat, sizeof(float) * 9);
}

void Batch::Flush() 
{
	// Use shader...
	m_material->Use();

	// Bind buffers.
	m_mesh->Bind();

	// Update mesh instance buffer...
	glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(Instance) * m_nInstanceCount, m_instances);

	// Draw...
	glDrawElementsInstanced(GL_TRIANGLES, m_mesh->IndexCount(), GL_UNSIGNED_INT, 0, m_nInstanceCount);

	// Unbind buffers...
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	m_nInstanceCount = 0;
}

void Batch::SetData() 
{
	// Vertices...
	m_vertices[0].m_v4Position = NVZMathLib::Vector4(-0.5f, -0.5f, 0.5f, 1.0f); // Front top left 0
	m_vertices[0].m_v4TexCoords = NVZMathLib::Vector4(0.0f, 0.0f, 0.0f, 1.0f);

	m_vertices[1].m_v4Position = NVZMathLib::Vector4(0.5f, -0.5f, 0.5f, 1.0f); // Front top right 1
	m_vertices[1].m_v4TexCoords = NVZMathLib::Vector4(1.0f, 0.0f, 0.0f, 1.0f);

	m_vertices[2].m_v4Position = NVZMathLib::Vector4(-0.5f, 0.5f, 0.5f, 1.0f); // Front bottom left 2
	m_vertices[2].m_v4TexCoords = NVZMathLib::Vector4(0.0f, 1.0f, 0.0f, 1.0f);

	m_vertices[3].m_v4Position = NVZMathLib::Vector4(0.5f, 0.5f, 0.5f, 1.0f); // Front bottom right 3
	m_vertices[3].m_v4TexCoords = NVZMathLib::Vector4(1.0f, 1.0f, 0.0f, 1.0f);

	m_vertices[4].m_v4Position = NVZMathLib::Vector4(-0.5f, -0.5f, -0.5f, 1.0f); // Back top left 4
	m_vertices[4].m_v4TexCoords = NVZMathLib::Vector4(0.0f, 0.0f, 0.0f, 1.0f);

	m_vertices[5].m_v4Position = NVZMathLib::Vector4(0.5f, -0.5f, -0.5f, 1.0f); // Back top right 5
	m_vertices[5].m_v4TexCoords = NVZMathLib::Vector4(1.0f, 0.0f, 0.0f, 1.0f);

	m_vertices[6].m_v4Position = NVZMathLib::Vector4(-0.5f, 0.5f, -0.5f, 1.0f); // Back bottom left 6
	m_vertices[6].m_v4TexCoords = NVZMathLib::Vector4(0.0f, 1.0f, 0.0f, 1.0f);

	m_vertices[7].m_v4Position = NVZMathLib::Vector4(0.5f, 0.5f, -0.5f, 1.0f); // Back bottom right 7
	m_vertices[7].m_v4TexCoords = NVZMathLib::Vector4(1.0f, 1.0f, 0.0f, 1.0f);


	unsigned short indices[36] = 
	{
		0, 1, 2, // Front face
		3, 2, 1,

		4, 0, 6, // Left face
		2, 6, 0,

		5, 4, 7, // Back face
		6, 7, 4,

		1, 0, 5, // Top face
		4, 5, 0,

		1, 5, 3, // Right face
		7, 3, 5,

		2, 7, 6, // Bottom face
		7, 2, 3
	};

	memcpy_s(m_indices, sizeof(unsigned short) * 36, indices, sizeof(unsigned short) * 36);
}

void Batch::CreateBuffers() 
{
	//// Create VAO, VBO, and Index buffers.
	//glGenVertexArrays(1, &m_glVAOHandle);
	//
	//// Bind vertex array to have all other buffers fall under it.
	//glBindVertexArray(m_glVAOHandle);
	//
	//glGenBuffers(1, &m_glVBOHandle);
	//glGenBuffers(1, &m_glInsHandle);
	//
	//glGenBuffers(1, &m_glEBOHandle);
	//
	//// Bind and fill buffers...
	//
	//// Model...
	//glBindBuffer(GL_ARRAY_BUFFER, m_glVBOHandle);
	//glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex) * 8, m_vertices, GL_STATIC_DRAW);
	//
	//// Index Buffer...
	//glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_glEBOHandle);
	//glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned short) * 36, m_indices, GL_STATIC_DRAW);
	//
	//// -----------------------------------------------------------------------------------------
	//// Vertex attributes
	//
	//// Model VBO
	//
	//glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0); // Position
	//glEnableVertexAttribArray(0);
	//
	//glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(sizeof(float) * 4)); // Tex Coords
	//glEnableVertexAttribArray(1);
	//
	//// Instance Buffer
	//
	//// Bind and fill...
	//glBindBuffer(GL_ARRAY_BUFFER, m_glInsHandle);
	//glBufferData(GL_ARRAY_BUFFER, sizeof(Instance) * MAX_INSTANCE_COUNT, m_instances, GL_STATIC_DRAW);
	//
	//// Attributes...
	//
	//// Color
	//glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(Instance), (void*)0);
	//glEnableVertexAttribArray(2);
	//
	//glVertexAttribDivisor(2, 1);
	//
	//// Model matrix
	//glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, sizeof(Instance), (void*)(sizeof(float) * 4));
	//glEnableVertexAttribArray(3);
	//
	//glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, sizeof(Instance), (void*)(sizeof(float) * 8));
	//glEnableVertexAttribArray(4);
	//
	//glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, sizeof(Instance), (void*)(sizeof(float) * 12));
	//glEnableVertexAttribArray(5);
	//
	//glVertexAttribPointer(6, 4, GL_FLOAT, GL_FALSE, sizeof(Instance), (void*)(sizeof(float) * 16));
	//glEnableVertexAttribArray(6);
	//
	//glVertexAttribDivisor(3, 1);
	//glVertexAttribDivisor(4, 1);
	//glVertexAttribDivisor(5, 1);
	//glVertexAttribDivisor(6, 1);
	//
	//// -----------------------------------------------------------------------------------------
	//
	//// Finished, unbind buffers.
	//glBindVertexArray(0);
	//glBindBuffer(GL_ARRAY_BUFFER, 0);
	//glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}
