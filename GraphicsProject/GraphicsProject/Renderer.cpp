#include "Renderer.h"
#include "glad/glad.h"
#include "glfw3.h"
#include "Mesh.h"
#include "Batch.h"
#include "Texture.h"
#include "Shader.h"
#include "glm.hpp"

using namespace NVZMathLib;

Renderer::Renderer(GLFWwindow* window) 
{
	m_window = window;
	m_pointLightShader = nullptr;

	// -----------------------------------------------------------------------------------------
	// Basic

	// Make window context current.
	glfwMakeContextCurrent(m_window);

	// Load OpenGL library using GLAD
	gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

	// Get window dimensions.
	glfwGetWindowSize(m_window, &m_nWindowWidth, &m_nWindowHeight);

	// Set up viewport
	glViewport(0, 0, m_nWindowWidth, m_nWindowHeight);

	// Set framebuffer clear color to black.
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	// Enable depth testing...
	glEnable(GL_DEPTH_TEST);

	// Enable back face culling...
	glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

	// Blend function for deferred shading
	glBlendFunci(0, GL_ONE, GL_ONE);

	// -----------------------------------------------------------------------------------------
    // Light volume sphere
	m_lightVolMesh = new Mesh("Assets/Primitives/sphere.obj");

	// -----------------------------------------------------------------------------------------
	// Quad and light buffers...

	CreateBuffers();

	// -----------------------------------------------------------------------------------------
	// Uniform blocks

	glGenBuffers(1, &m_glUBOMatrixHandle);
	glBindBuffer(GL_UNIFORM_BUFFER, m_glUBOMatrixHandle);

	// Map buffer range...
	glBindBufferRange(GL_UNIFORM_BUFFER, 0, m_glUBOMatrixHandle, 0, sizeof(ViewProjBlock));

	// Initialize view matrix...
	m_matrices.m_v3ViewPos = glm::vec3(0.0f, 1.0f, 3.0f);

	m_matrices.m_viewMat = glm::lookAt<float>(m_matrices.m_v3ViewPos, m_matrices.m_v3ViewPos + glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, 1.0f, 0.0f));
}

Renderer::~Renderer() 
{
	delete m_lightVolMesh;

	glDeleteBuffers(1, &m_glUBOMatrixHandle);
	
	glDeleteVertexArrays(1, &m_glQuadVAO);
	glDeleteBuffers(1, &m_glQuadEBO);
	glDeleteBuffers(1, &m_glQuadVBO);
}

void Renderer::AddBatch(Batch* batch) 
{
	m_batches.Push(batch);
}

void Renderer::Start() 
{
	glEnable(GL_DEPTH_TEST);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Projection matrix...
	m_matrices.m_projMat = glm::perspective<float>(glm::radians(FIELD_OF_VIEW), static_cast<float>(m_nWindowWidth) / static_cast<float>(m_nWindowHeight), 0.1f, 1000.0f);

	// Send matrices to GPU...
	glBindBuffer(GL_UNIFORM_BUFFER, m_glUBOMatrixHandle);

	glBufferData(GL_UNIFORM_BUFFER, sizeof(ViewProjBlock), &m_matrices, GL_STATIC_DRAW);

	glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

void Renderer::ClearFramebuffer() 
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void Renderer::DrawFinal() 
{
	for (int i = 0; i < m_batches.Count(); ++i)
		m_batches[i]->Flush();
}

void Renderer::End() 
{
	glfwSwapBuffers(m_window);
}

void Renderer::DrawFSQuad(Shader* shader, Texture** textures, int nTextureCount) 
{
	// Use shader...
	shader->Use();

	// Bind all textures...
	for (int i = 0; i < nTextureCount; ++i)
	{
		glActiveTexture(GL_TEXTURE0 + i);
		glBindTexture(GL_TEXTURE_2D, textures[i]->GetHandle());
	}
	
	// Reset active texture.
	glActiveTexture(GL_TEXTURE0);

	// Bind quad...
	glBindVertexArray(m_glQuadVAO);
	glBindBuffer(GL_ARRAY_BUFFER, m_glQuadVBO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_glQuadEBO);

	// Draw...
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}

void Renderer::BindTextures(Texture** textures, int nTextureCount) 
{
	for (int i = 0; i < nTextureCount; ++i)
	{
		glActiveTexture(GL_TEXTURE0 + i);
		glBindTexture(GL_TEXTURE_2D, textures[i]->GetHandle());
	}
}

void Renderer::BindFSQuad() 
{
	// Bind quad...
	glBindVertexArray(m_glQuadVAO);
	glBindBuffer(GL_ARRAY_BUFFER, m_glQuadVBO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_glQuadEBO);
}

void Renderer::UnbindFSQuad() 
{
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void Renderer::DrawFSQuadNoState() 
{
	// Draw...
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}

void Renderer::ResetFramebufferBinding()
{
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void Renderer::SetPLightShader(Shader* lightShader) 
{
	m_pointLightShader = lightShader;
}

void Renderer::SetDLightShader(Shader* lightShader)
{
	m_directionalLightShader = lightShader;
}

int Renderer::AddPointLight(Vector4 v4Color, Vector3 v3Position, float fRadius) 
{
	if (m_nPLightCount >= MAX_LIGHT_COUNT)
		return 0;

	m_pointLights[m_nPLightCount].m_v4Color = { v4Color.x, v4Color.y, v4Color.z, fRadius };
	
	// Create and translate matrix to the correct coordinates.
	glm::mat4 lightMat = glm::translate(glm::mat4(), glm::vec3(v3Position.x, v3Position.y, v3Position.z));

	// Scale light matrix by radius...
	lightMat = glm::scale(lightMat, glm::vec3(fRadius));

	// Copy matrix into instance and increment light count...
	memcpy_s(m_pointLights[m_nPLightCount].m_modelMat, sizeof(float) * 16, glm::value_ptr(lightMat), sizeof(float) * 16);

	return m_nPLightCount++;
}

void Renderer::TransformPointLight(int nIndex, NVZMathLib::Vector3 v3Position, float fRadius) 
{
	PointLight& selectedLight = m_pointLights[nIndex];

	// Scale light by radius...
	glm::mat4 lightMat = glm::scale(glm::mat4(), glm::vec3(fRadius));

	// Set light color w to radius.
	selectedLight.m_v4Color.w = fRadius;

	// Translate light from origin to new position.
	lightMat[3] = glm::vec4(v3Position.x, v3Position.y, v3Position.z, 1.0f);

	// Copy new matrix.
	memcpy_s(selectedLight.m_modelMat, sizeof(float) * 16, glm::value_ptr(lightMat), sizeof(float) * 16);
}

void Renderer::TransformPointLight(int nIndex, NVZMathLib::Vector3 v3Position)
{
	PointLight& selectedLight = m_pointLights[nIndex];

	// Set light position.
	selectedLight.m_modelMat[12] = v3Position.x;
	selectedLight.m_modelMat[13] = v3Position.y;
	selectedLight.m_modelMat[14] = v3Position.z;
}

void Renderer::RemovePointLight(int nIndex) 
{
	int nSize = MAX_LIGHT_COUNT - nIndex - 1;

	// Remove array gap...
	memcpy_s(&m_pointLights[nIndex], nSize, &m_pointLights[nIndex + 1], nSize - 1);

	--m_nPLightCount;
}

void Renderer::RunDeferredDirectionalLight(Texture** textures, int nTextureCount) 
{
	// Enable blending...
	glEnable(GL_BLEND);

	if (!m_directionalLightShader)
		return;

	// Use shader...
	m_directionalLightShader->Use();

	m_directionalLightShader->SetUniformVec3("lightDir", Vector3(1.0f, -0.1f, 0.0f).Normalised());
	m_directionalLightShader->SetUniformVec3("lightColor", Vector3(1.0f, 1.0f, 1.0f));

	// Bind all textures...
	for (int i = 0; i < nTextureCount; ++i)
	{
		glActiveTexture(GL_TEXTURE0 + i);
		glBindTexture(GL_TEXTURE_2D, textures[i]->GetHandle());
	}

	// Reset active texture.
	glActiveTexture(GL_TEXTURE0);

	// Bind quad...
	glBindVertexArray(m_glQuadVAO);
	glBindBuffer(GL_ARRAY_BUFFER, m_glQuadVBO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_glQuadEBO);

	// Draw...
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}

void Renderer::RunDeferredPointLighting(Texture** textures, int nTextureCount)
{
	// Disable depth testing...
	glDisable(GL_DEPTH_TEST);
	glCullFace(GL_FRONT);

	if (!m_pointLightShader)
		return;

	m_pointLightShader->Use();

	// Bind all textures...
	for (int i = 0; i < nTextureCount; ++i)
	{
		glActiveTexture(GL_TEXTURE0 + i);
		glBindTexture(GL_TEXTURE_2D, textures[i]->GetHandle());
	}

	// Bind buffers...
	m_lightVolMesh->Bind();

	// Update lights...
	glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(PointLight) * m_nPLightCount, m_pointLights);

	// Draw...
	glDrawElementsInstanced(GL_TRIANGLES, m_lightVolMesh->IndexCount(), GL_UNSIGNED_INT, 0, m_nPLightCount);

	// Reset bindings...
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	// Disable blending for the next frame.
	glDisable(GL_BLEND);
	glCullFace(GL_BACK);
}

void Renderer::ReportErrors() 
{
	// Get first error code.
	int error = glGetError();

	// If & while the error code is not 0...
	while(error) 
	{
		// Print to console.
		std::cout << "OpenGL error occured, code: " << error << std::endl;

		// Fetch next error.
		error = glGetError();
	}
}

void Renderer::SetViewMatrix(float* fMatrix) 
{
	memcpy_s(&m_matrices.m_viewMat, sizeof(float) * 16, fMatrix, sizeof(float) * 16);
	m_matrices.m_v3ViewPos = { -fMatrix[11], -fMatrix[12], -fMatrix[13] };
}

void Renderer::SetViewMatrix(glm::mat4 matrix, glm::vec3 v3ViewPos)
{
	m_matrices.m_viewMat = matrix;
	m_matrices.m_v3ViewPos = v3ViewPos;
}

int Renderer::WindowWidth() 
{
	return m_nWindowWidth;
}

int Renderer::WindowHeight() 
{
	return m_nWindowHeight;
}

void Renderer::CreateBuffers() 
{
	// Generate buffers...
	glGenBuffers(1, &m_glQuadVBO);
	glGenBuffers(1, &m_glQuadEBO);
	glGenVertexArrays(1, &m_glQuadVAO);

	// Bind VAO...
	glBindVertexArray(m_glQuadVAO);

	// Vertices... x, y = positions, z, w = tex coords.
	NVZMathLib::Vector4 v2Vertices[4] = 
	{
		NVZMathLib::Vector4(-1.0f, -1.0f, 0.0f, 0.0f),
		NVZMathLib::Vector4(1.0f, -1.0f, 1.0f, 0.0f),
		NVZMathLib::Vector4(-1.0f, 1.0f, 0.0f, 1.0f),
		NVZMathLib::Vector4(1.0f, 1.0f, 1.0f, 1.0f)
	};

	// Bind and fill VBO
	glBindBuffer(GL_ARRAY_BUFFER, m_glQuadVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 16, v2Vertices, GL_STATIC_DRAW);

	// Attributes...

	// Positions
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 4, (void*)0);
	glEnableVertexAttribArray(0);

	// Texture Coordinates
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 4, (void*)(sizeof(float) * 2));
	glEnableVertexAttribArray(1);

	// Indices...
	unsigned int indices[6] =
	{
		0, 1, 2,
		3, 2, 1
	};

	// Bind and fill index buffer...
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_glQuadEBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * 6, indices, GL_STATIC_DRAW);

	// Unbind buffers.
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}