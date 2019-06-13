#include "RenderSingle.h"
#include "glad\glad.h"
#include "Material.h"
#include "Mesh.h"
#include "glm.hpp"

RenderSingle::RenderSingle(Mesh* mesh, Material* material) : RenderObject(mesh, material) 
{

}

RenderSingle::~RenderSingle()
{
	
}

void RenderSingle::Draw() 
{
	// Bind material...
	m_material->Use();

	// Bind mesh...
	m_mesh->Bind();

	// Update single instance buffer.
	glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(Instance), &m_instance);

	// Draw...
	glDrawElements(GL_TRIANGLES, m_mesh->IndexCount(), GL_UNSIGNED_INT, 0);

	// Unbind buffers...
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void RenderSingle::UpdateObject(float* modelMatrix, NVZMathLib::Vector4 v4Color) 
{
	// Copy model matrix...
	memcpy_s(m_instance.m_modelMat, sizeof(float) * 16, modelMatrix, sizeof(float) * 16);

	// Construct normal matrix...
	glm::mat3 normalMat =
	{
		glm::vec3(modelMatrix[0], modelMatrix[1], modelMatrix[2]),
		glm::vec3(modelMatrix[4], modelMatrix[5], modelMatrix[6]),
		glm::vec3(modelMatrix[8], modelMatrix[9], modelMatrix[10])
	};

	// Copy normal matrix...
	memcpy_s(m_instance.m_normalMat, sizeof(float) * 9, &normalMat, sizeof(float) * 9);
}