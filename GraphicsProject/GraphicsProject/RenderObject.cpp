#include "RenderObject.h"

RenderObject::RenderObject(Mesh* mesh, Material* material)
{
	m_mesh = mesh;
	m_material = material;
}

RenderObject::~RenderObject() 
{

}