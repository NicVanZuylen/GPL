#pragma once
#include "DynamicArray.h"

class Shader;
class Texture;
class MeshRenderer;
class StaticMeshRenderer;

class Material 
{
public:

	Material(Shader* shader);

	~Material();

	/*
	Description: Add a texture map for use in this material's shader.
	Param:
	    Texture* map: The texture map to be added. Material DOES NOT take OWNERSHIP of the texture memory.
	*/
	void AddMap(Texture* map);

	/*
	Description: Remove a texture map from this material.
	Param:
	    Texture* map: The pointer to the texture map to be removed.
	*/
	void RemoveMap(Texture* map);

	/*
	Description: Remove a texture map from this material.
	Param:
	    int nMapIndex: The index of the texture map to remove.
	*/
	void RemoveMap(int nMapIndex);

	/*
	Description: Get the amount of maps used by this material.
	Return Type: int
	*/
	int MapCount();

	/*
	Description: Bind the shader and texture maps of this material.
	*/
	void Use();

	/*
	Description: Draw all meshes using this material without texture and shader state changes.
	*/
	void DrawMeshes();

	/*
	Description: Draw static meshes for this material.
	*/
	void DrawStaticMeshes();

	/*
	Description: Set the shader used by this material for rendering.
	Param:
		Shader* shader: The new shader to use.
	*/
	void SetShader(Shader* shader);

	/*
	Description: Returns the shader of this material.
	Return Type: Shader*
	*/
	Shader* GetShader();

	/*
	Description: Get the array of all meshes using this material.
	Return Type: DynamicArray<MeshRenderer*>&
	*/
	DynamicArray<MeshRenderer*>& GetMeshes();

	/*
	Description: Get the array of all static meshes using this material.
	Return Type: DynamicArray<StaticMeshRenderer*>&
	*/
	DynamicArray<StaticMeshRenderer*>& GetStaticMeshes();

	void SetAmbientColor(float r, float g, float b);

	void SetDiffuseColor(float r, float g, float b);

	void SetSpecularColor(float r, float g, float b);

	void SetEmissionColor(float r, float g, float b);

	void SetSpecularShininess(const float& fShininess);

	void SetAlpha(float fAlpha);

private:

	DynamicArray<MeshRenderer*> m_meshes;
	DynamicArray<StaticMeshRenderer*> m_staticMeshes;

	Shader* m_shader;
	DynamicArray<Texture*> m_maps;

	float m_fAmbientColor[3];
	float m_fDiffuseColor[3];
	float m_fSpecularColor[3];
	float m_fEmissiveColor[3];
	float m_fSpecShininess;
	float m_fAlpha;
};