#pragma once
#include "Vector4.h"

class Mesh;
class Material;

class RenderObject 
{
public:

	RenderObject(Mesh* mesh, Material* material);

	virtual ~RenderObject();

	virtual void Draw() = 0;

protected:

	struct Instance
	{
		NVZMathLib::Vector4 m_v4Color;
		float m_modelMat[16];
		float m_normalMat[9];
	};

	Mesh* m_mesh;
	Material* m_material;
};
