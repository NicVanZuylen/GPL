#include "MeshRenderer.h"
#include "Mesh.h"
#include "Material.h"
#include "Shader.h"
#include "GLAD\glad.h"
#include "glm.hpp"
#include "glm\include\ext.hpp"

using namespace NVZMathLib;

MeshRenderer::MeshRenderer(Mesh* mesh, Material* material, int nMaxInstances) 
{
	m_mesh = mesh;
	m_nMaxInstances = nMaxInstances;

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

	// Construct VAO for this renderer.

	// Generate and bind VAO...
	glGenVertexArrays(1, &m_glVAOHandle);
	glBindVertexArray(m_glVAOHandle);

	// Bind Index buffer to this VAO...
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_mesh->IndexBufferHandle());

	// Bind vertex buffer to this VAO...
	glBindBuffer(GL_ARRAY_BUFFER, m_mesh->VBOHandle());

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

	// Generate and bind instance buffer...
	glGenBuffers(1, &m_glInsHandle);
	glBindBuffer(GL_ARRAY_BUFFER, m_glInsHandle);

	// Fill with empty single instance data.
	glBufferData(GL_ARRAY_BUFFER, sizeof(Instance) * 1, 0, GL_DYNAMIC_DRAW);

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

MeshRenderer::~MeshRenderer() 
{
	glDeleteBuffers(1, &m_glInsHandle);
	glDeleteVertexArrays(1, &m_glVAOHandle);

	if(m_material)
	    m_material->GetMeshes().PopAt(m_nMaterialIndex);
}

void MeshRenderer::Draw() 
{
	// Material should be bound externally.

	// Bind buffers.
	glBindVertexArray(m_glVAOHandle);
	glBindBuffer(GL_ARRAY_BUFFER, m_mesh->VBOHandle());
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_mesh->IndexBufferHandle());
	glBindBuffer(GL_ARRAY_BUFFER, m_glInsHandle);

	// Update mesh instance buffer...
	glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(Instance) * m_instances.Count(), m_instances.Data());

	// Draw...
	glDrawElementsInstanced(GL_TRIANGLES, m_mesh->IndexCount(), GL_UNSIGNED_INT, 0, m_instances.Count());

	// Unbind buffers...
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

int MeshRenderer::AddInstance() 
{
	int nInstanceCount = m_instances.Count();

	// Do not add if the max instance count has been exceeded.
	if (nInstanceCount >= m_nMaxInstances)
		return -1;

	Instance newInstance;

	int nSize = m_instances.GetSize();

	m_instances.Push(newInstance);

	// Resize instance buffer if the new instance count exceeds the buffer size...
	if (nSize < nInstanceCount + 1)
	{
		glBindBuffer(GL_ARRAY_BUFFER, m_glInsHandle);
		glBufferData(GL_ARRAY_BUFFER, sizeof(Instance) * m_instances.Count(), 0, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	return nInstanceCount;
}

void MeshRenderer::UpdateInstance(const int& nIndex, float* modelMatrix)
{
	Instance& selectedInstance = m_instances[nIndex];

	// Copy model matrix.
	memcpy_s(selectedInstance.m_modelMat, sizeof(float) * 16, modelMatrix, sizeof(float) * 16);

	// Construct normal matrix.
	glm::mat3 normalMat =
	{
		glm::vec3(modelMatrix[0], modelMatrix[1], modelMatrix[2]),
		glm::vec3(modelMatrix[4], modelMatrix[5], modelMatrix[6]),
		glm::vec3(modelMatrix[8], modelMatrix[9], modelMatrix[10])
	};

	// Copy normal matrix into instance...
	memcpy_s(selectedInstance.m_normalMat, sizeof(float) * 9, &normalMat, sizeof(float) * 9);
}

void MeshRenderer::UpdateInstance(const int& nIndex, float* modelMatrix, const NVZMathLib::Vector4& v4Color) 
{
	Instance& selectedInstance = m_instances[nIndex];

	// Copy color.
	selectedInstance.m_v4Color = v4Color;

	// Copy model matrix.
	memcpy_s(selectedInstance.m_modelMat, sizeof(float) * 16, modelMatrix, sizeof(float) * 16);

	// Construct normal matrix.
	glm::mat3 normalMat =
	{
		glm::vec3(modelMatrix[0], modelMatrix[1], modelMatrix[2]),
		glm::vec3(modelMatrix[4], modelMatrix[5], modelMatrix[6]),
		glm::vec3(modelMatrix[8], modelMatrix[9], modelMatrix[10])
	};

	// Copy normal matrix into instance...
	memcpy_s(selectedInstance.m_normalMat, sizeof(float) * 9, &normalMat, sizeof(float) * 9);
}

void MeshRenderer::RemoveInstance(const int& nIndex) 
{
	m_instances.PopAt(nIndex);
}

void MeshRenderer::SetMaterial(Material* material) 
{
	// Remove from old material...
	if(m_material)
	    m_material->GetMeshes().PopAt(m_nMaterialIndex);

	m_material = material;

	// Add to new material if it is not null.
	if (m_material)
	{
		m_nMaterialIndex = m_material->GetMeshes().Count();
		m_material->GetMeshes().Push(this);
	}
	else
		m_nMaterialIndex = 0;
}