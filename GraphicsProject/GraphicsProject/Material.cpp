#include "Material.h"
#include "glad\glad.h"
#include "Shader.h"
#include "Texture.h"
#include "MeshRenderer.h"
#include "StaticMeshRenderer.h"
#include <iostream>

Material::Material(Shader* shader) 
{
	m_shader = shader;
}

Material::~Material() 
{
	m_shader = nullptr;

	// Remove all mesh renderers...
	for (int i = 0; i < m_meshes.Count(); ++i)
		m_meshes[i]->SetMaterial(nullptr);
}

void Material::AddMap(Texture* map) 
{
	// Don't allow maps to be added if there is already 16 maps active.
	if (m_maps.Count() < 16)
		m_maps.Push(map);
}

void Material::RemoveMap(Texture* map) 
{
	for(int i = 0; i < m_maps.Count(); ++i) 
	{
		if(map == m_maps[i]) 
		{
			m_maps.PopAt(i);
			return;
		}
	}
}

void Material::RemoveMap(int nMapIndex) 
{
	m_maps.PopAt(nMapIndex);
}

int Material::MapCount() 
{
	return m_maps.Count();
}

void Material::Use() 
{
	m_shader->Use();

	// Bind all texture maps...
	for(int i = 0; i < m_maps.Count(); ++i) 
	{
		glActiveTexture(GL_TEXTURE0 + i);
		glBindTexture(GL_TEXTURE_2D, m_maps[i]->GetHandle());
	}

	// Reset active texture.
	glActiveTexture(GL_TEXTURE0);
}

void Material::DrawMeshes() 
{
	// Use this material for drawing...
	Use();

	// Draw all associated meshes.
	for (int i = 0; i < m_meshes.Count(); ++i)
		m_meshes[i]->Draw();
}

void Material::DrawStaticMeshes() 
{
	// Use this material for drawing...
	Use();

	// Draw all associated static meshes.
	for (int i = 0; i < m_staticMeshes.Count(); ++i)
		m_staticMeshes[i]->Draw();
}

void Material::SetShader(Shader* shader)
{
	m_shader = shader;
}

Shader* Material::GetShader() 
{
	return m_shader;
}

DynamicArray<MeshRenderer*>& Material::GetMeshes() 
{
	return m_meshes;
}

DynamicArray<StaticMeshRenderer*>& Material::GetStaticMeshes()
{
	return m_staticMeshes;
}

void Material::SetAmbientColor(float r, float g, float b)
{
	m_fAmbientColor[0] = r;
	m_fAmbientColor[1] = g;
	m_fAmbientColor[2] = b;
}

void Material::SetDiffuseColor(float r, float g, float b) 
{
	m_fDiffuseColor[0] = r;
	m_fDiffuseColor[1] = g;
	m_fDiffuseColor[2] = b;
}

void Material::SetSpecularColor(float r, float g, float b) 
{
	m_fSpecularColor[0] = r;
	m_fSpecularColor[1] = g;
	m_fSpecularColor[2] = b;
}

void Material::SetEmissionColor(float r, float g, float b) 
{
	m_fEmissiveColor[0] = r;
	m_fEmissiveColor[1] = g;
	m_fEmissiveColor[2] = b;
}

void Material::SetSpecularShininess(const float& fShininess) 
{
	m_fSpecShininess = fShininess;

	if(m_shader) 
	{
		m_shader->Use();
		m_shader->SetUniformFloat("specularShininess", m_fSpecShininess);
		glUseProgram(0);
	}
}

void Material::SetAlpha(float fAlpha) 
{
	m_fAlpha = fAlpha;
}