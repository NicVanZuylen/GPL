#pragma once
#include "RenderObject.h"

class RenderSingle : public RenderObject
{
public:

	RenderSingle(Mesh* mesh, Material* material);

	virtual ~RenderSingle();

	void Draw() override;

	void UpdateObject(float* modelMatrix, NVZMathLib::Vector4 v4Color = { 1.0f, 1.0f, 1.0f, 1.0f });

private:

	Instance m_instance;
};