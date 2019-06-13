#pragma once
#include "Vector4.h"
#include "DynamicArray.h"

class Mesh;
class Material;

class MeshRenderer 
{
public:

	MeshRenderer(Mesh* mesh, Material* material, int nMaxInstances = 512);

	virtual ~MeshRenderer();

	virtual void Draw();

	/*
	Description: Add a new mesh instance to be renderered with this renderer's material, returns an index used for accessing the instance for modification.
	-1 is returned when adding an instance if the max instance count has already been reached.
	Return Type: int
	*/
	int AddInstance();

	/*
	Description: Update an existing instance at the provided index.
	Param:
	    const int& nIndex: The index of the instance.
	    float* modelMatrix: The model matrix in float ptr format.
	*/
	void UpdateInstance(const int& nIndex, float* modelMatrix);

	/*
	Description: Update an existing instance at the provided index.
	Param:
	    const int& nIndex: The index of the instance.
		float* modelMatrix: The model matrix in float ptr format.
		const Vector4& v4Color: The new tint color of meshes rendered with this renderer (if applicable).
	*/
	void UpdateInstance(const int& nIndex, float* modelMatrix, const NVZMathLib::Vector4& v4Color);

	/*
	Description: Remove an existing instance at the provided index.
	Param:
	    const int& nIndex: The index to remove from.
	*/
	void RemoveInstance(const int& nIndex);

	/*
	Description: Set the material used for rendering this mesh instance batch.
	Param:
	    Material* material: The material to use.
	*/
	void SetMaterial(Material* material);

private:

	struct Instance
	{
		NVZMathLib::Vector4 m_v4Color;
		float m_modelMat[16];
		float m_normalMat[9];
	};

	unsigned int m_glVAOHandle;
	unsigned int m_glInsHandle;

	DynamicArray<Instance> m_instances;

	Mesh* m_mesh;
	Material* m_material;
	int m_nMaxInstances;
	int m_nMaterialIndex; // The index of this mesh renderer in it's materials mesh renderer array.
};