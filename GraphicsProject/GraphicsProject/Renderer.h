#pragma once
#include "DynamicArray.h"
#include "Vector4.h"
#include "Matrix4.h"
#include "glm.hpp"
#include "gtc/matrix_transform.hpp"
#include "gtc/type_ptr.hpp"

struct GLFWwindow;

class Shader;
class Texture;
class Mesh;
class Batch;

#define FIELD_OF_VIEW 45.0f

#define MAX_LIGHT_COUNT 512

class Renderer 
{
public:

	Renderer(GLFWwindow* window);

	~Renderer();

	/*
	Description: Add a batch to be draw by the renderer automatically.
	Param:
	    Batch* batch: The batch to be rendered. 
	*/
	void AddBatch(Batch* batch);

	/*
	Description: Begin the rendering process for this frame...
	*/
	void Start();

	/*
	Description: Clears the currently bound framebuffer to the render clear color.
	*/
	void ClearFramebuffer();

	/*
	Draw any final undrawn objects added to batches.
	*/
	void DrawFinal();

	/*
	Description: End the rendering process for this frame.
	*/
	void End();

	/*
	Description: Draw a fullscreen quad for use in post effects and deferred shading.
	Param:
	    Shader* shader: The shader to be used in drawing the quad.
		Texture** textures: Array of texture pointers for textures used in drawing the quad.
		int nTextureCount: The amount of textures present in the array.
	*/
	void DrawFSQuad(Shader* shader, Texture** textures, int nTextureCount);

	/*
	Description: Bind all provided textures...
	Param:
	    Texture** textures: Pointer to an array of texture pointers, used for binding the textures they reference.
		int nTextureCount: The amount of textures in the array to bind.
	*/
	void BindTextures(Texture** textures, int nTextureCount);

	/*
	Description: Bind the built-in fullscreen quad for rendering.
	*/
	void BindFSQuad();

	/*
	Description: Un-bind the built-in fullscreen quad.
	*/
	void UnbindFSQuad();

	/*
	Description: Draw a fullscreen quad using the currently bound textures and shader.
	*/
	void DrawFSQuadNoState();

	/*
	Description: Resets the framebuffer binding to the default window framebuffer.
	*/
	void ResetFramebufferBinding();

	/*
	Description: Set the shader used for deferred shading with point lights.
	Param:
	    Shader* lightShader: The shader to be used for shading with point lights. 
	*/
	void SetPLightShader(Shader* lightShader);

	/*
	Description: Set the shader used for deferred shading with directional lights.
	Param:
		Shader* lightShader: The shader to be used for shading with directional lights.
	*/
	void SetDLightShader(Shader* lightShader);

	/*
	Description: Add a light for use in the deferred shading pass... The returned value is the index used to select lights for removal or modification.
	Return Type: int
	Param:
	    Vector4 v4Color: The color of the light to be added.
		Vector3 v3Position: The worldspace position of the light to be added.
		float fRadius: The effective radius of the light to be added.
	*/
	int AddPointLight(NVZMathLib::Vector4 v4Color, NVZMathLib::Vector3 v3Position, float fRadius);


	/*
	Description: Modify the position and radius of a light volume at the specified index.
	Param:
	    int nIndex: The index of the light instance to modify.
	    Vector3 v3Position: The new position of the light volume.
	    float fRadius: The new radius of the light.
	*/
	void TransformPointLight(int nIndex, NVZMathLib::Vector3 v3Position, float fRadius);

	/*
	Description: Modify the position and radius of a light volume at the specified index.
	Param:
	    int nIndex: The index of the light instance to modify.
		Vector3 v3Position: The new position of the light volume.
		float fRadius: The new radius of the light.
	*/
	void TransformPointLight(int nIndex, NVZMathLib::Vector3 v3Position);

	/*
	Description: Remove a light from the scene.
	Param:
	    int nIndex: The index of the light in the light array.
	*/
	void RemovePointLight(int nIndex);

	/*
	Description: Run deferred shading for directional lighting.
	Param:
	    Texture** textures: Array of texture pointers for textures used in drawing the quad.
		int nTextureCount: The amount of textures present in the array.
	*/
	void RunDeferredDirectionalLight(Texture** textures, int nTextureCount);

	/*
	Description: Run deferred shading for all point lights in the scene.
	Param:
	    Texture** textures: Array of texture pointers for textures used in drawing the quad.
	    int nTextureCount: The amount of textures present in the array.
	*/
	void RunDeferredPointLighting(Texture** textures, int nTextureCount);

	/*
	Description: Print all OpenGL error codes to the console if any errors occur.
	*/
	static void ReportErrors();

	/*
	Description: Set the camera view matrix.
	Param:
	    float* fMatrix: Pointer to the raw float matrix.
	*/
	void SetViewMatrix(float* fMatrix);

	/*
	Description: Set the camera view matrix.
	Param:
		mat4 matrix: The matrix to use.
		vec3 v3ViewPos: The position of the camera in worldspace.
	*/
	void SetViewMatrix(glm::mat4 matrix, glm::vec3 v3ViewPos);

	// Width of the borderless window in pixels.
	int WindowWidth();

	// Height of the borderless window in pixels.
	int WindowHeight();

private:

	void CreateBuffers();

	DynamicArray<Batch*> m_batches;

	GLFWwindow* m_window;
	int m_nWindowWidth;
	int m_nWindowHeight;

	// Camera
	glm::mat4 viewMat;

	// Quad VBO
	unsigned int m_glQuadVAO;
	unsigned int m_glQuadVBO;
	unsigned int m_glQuadEBO;

	// Matrix uniform buffer
	struct ViewProjBlock
	{
		glm::mat4 m_viewMat;
		glm::mat4 m_projMat;
		glm::vec3 m_v3ViewPos;
	};

	ViewProjBlock m_matrices;
	unsigned int m_glUBOMatrixHandle;

	struct PointLight
	{
		NVZMathLib::Vector4 m_v4Color;
		float m_modelMat[16];
		float m_normalMatPadding[9];
	};

	// Directional lighting shader
	Shader* m_directionalLightShader;
	NVZMathLib::Vector3 m_v3DirectionalLightDir;
	NVZMathLib::Vector3 m_v3DirectionalLightCol;

	// Point lighting shader
	Shader* m_pointLightShader;

	// Light array
	PointLight m_pointLights[MAX_LIGHT_COUNT];
	int m_nPLightCount;

	// Light buffer handles
	Mesh* m_lightVolMesh;
};