#ifndef MESH_H
#define MESH_H

#include "DynamicArray.h"
#include "Vector4.h"
#include <vector>

class Texture;
class Shader;
class Material;

enum ETextureMapType
{
	TEXTURE_MAP_DIFFUSE = 1,
	TEXTURE_MAP_NORMAL = 1 << 1,
	TEXTURE_MAP_SPECULAR = 1 << 2,
	TEXTURE_MAP_SPECULARHIGHLIGHT = 1 << 3,
	TEXTURE_MAP_DISPLACEMENT = 1 << 4,
	TEXTURE_MAP_ALPHA = 1 << 5,
	TEXTURE_MAP_AMBIENT = 1 << 6
};

class Mesh 
{
public:

	Mesh();

	Mesh(const char* szFilePath);

	~Mesh();

	/*
	Description: Load the mesh from a file, and any included materials.
	Param:
	    const char* szFilePath: The path to the .obj mesh file.
		unsigned int textureFlags: The texturemaps to load from the obj's materials, by default all maps are loaded.
	*/
	void Load(const char* szFilePath, unsigned int textureFlags = 0xFFFFFFFF);

	/*
	Description: Set the shader of the material at the provided index.
	Param:
	    Shader* shader: The shader to use.
	    int nMaterialIndex: The index of the material in the model object.
	*/
	void SetShader(Shader* shader, int nMaterialIndex);

	/*
	Description: Bind the VAO of this mesh for use in drawing.
	*/
	void Bind();

	/*
	Description: Get the VBO handle for this mesh.
	*/
	unsigned int VBOHandle();

	/*
	Description: Get the Index buffer handle for this mesh.
	*/
	unsigned int IndexBufferHandle();

	/*
	Description: Get the OpenGL handle for this mesh's vertex array object.
	Return Type: unsigned int
	*/
	unsigned int VAOHandle();

	/*
	Description: Get the handle of the instance buffer for this mesh.
	Return Type: unsigned int
	*/
	unsigned int InstanceHandle();

	/*
	Description: Get the amount of vertices in the entire mesh.
	Return Type: unsigned int
	*/
	unsigned int VertexCount();

	/*
	Description: Get the amount of indices in the entire mesh.
	Return Type: unsigned int
	*/
	unsigned int IndexCount();

	struct Vertex
	{
		NVZMathLib::Vector4 m_v4Position;
		NVZMathLib::Vector4 m_v4Normal;
		NVZMathLib::Vector4 m_v4Tangent;
		NVZMathLib::Vector2 m_v2TexCoords;
	};

private:

	void CalculateTangents(std::vector<Vertex>& vertices, std::vector<unsigned int>& indices);

	struct Instance
	{
		NVZMathLib::Vector4 m_v4Color;
		float m_modelMat[16];
		float m_normalMat[9];
	};

	const char* m_szFilePath;

	unsigned int m_glVAOHandle;
	unsigned int m_glVBOHandle;
	unsigned int m_glInsHandle;
	unsigned int m_glEBOHandle;

	unsigned int m_nWholeVertexCount;
	unsigned int m_nWholeIndexCount;

	// Mesh chunks
	struct MeshChunk 
	{
		// OpenGL handles
		unsigned int m_glVAOHandle;
		unsigned int m_glVBOHandle;
		unsigned int m_glEBOHandle;

		unsigned int m_nIndexCount;
		int m_nMaterialIndex;
	};

	MeshChunk* m_meshes;
	int m_nMeshChunkCount;

	Material** m_materials;
	int m_nMaterialCount;

	DynamicArray<Texture*> m_textureMaps;

	bool m_bEmptyMesh;
};

#endif /* MESH_H */