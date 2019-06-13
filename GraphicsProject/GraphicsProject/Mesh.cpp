#include "Mesh.h"
#include "GLAD\glad.h"
#include "Texture.h"
#include "Material.h"
#include "Batch.h"
#include <iostream>

// Using tiny obj loader header lib for .obj file loading.
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

Mesh::Mesh() 
{
	m_bEmptyMesh = true;
	m_materials = nullptr;
	m_szFilePath = nullptr;
}

Mesh::Mesh(const char* szFilePath) 
{
	m_bEmptyMesh = true;
	m_materials = nullptr;
	m_szFilePath = szFilePath;

	Load(szFilePath);

	m_bEmptyMesh = false;
}

Mesh::~Mesh() 
{
	if(!m_bEmptyMesh) 
	{
		for(int i = 0; i < m_nMeshChunkCount; ++i) 
		{
			glDeleteVertexArrays(1, &m_meshes[i].m_glVAOHandle);

			glDeleteBuffers(1, &m_meshes[i].m_glEBOHandle);
			glDeleteBuffers(1, &m_meshes[i].m_glVBOHandle);
		}

		m_nMeshChunkCount = 0;

		delete[] m_meshes;

		for(int i = 0; i < m_nMaterialCount; ++i) 
			delete m_materials[i];

		delete[] m_materials;

		for (int i = 0; i < m_textureMaps.Count(); ++i)
			delete m_textureMaps[i];
	}
}

void Mesh::Load(const char* szFilePath, unsigned int textureFlags) 
{
	// Delete old mesh if there is one.
	if(!m_bEmptyMesh) 
	{
		for (int i = 0; i < m_nMeshChunkCount; ++i)
		{
			glDeleteVertexArrays(1, &m_meshes[i].m_glVAOHandle);

			glDeleteBuffers(1, &m_meshes[i].m_glEBOHandle);
			glDeleteBuffers(1, &m_meshes[i].m_glVBOHandle);
		}
		
		m_nMeshChunkCount = 0;

		delete[] m_meshes;

		for (int i = 0; i < m_nMaterialCount; ++i)
			delete m_materials[i];

		delete[] m_materials;

		for (int i = 0; i < m_textureMaps.Count(); ++i)
			delete m_textureMaps[i];
	}

	m_szFilePath = szFilePath;

	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string errorMessage;
	
	std::string path = szFilePath;
	const char* szMaterialPath = path.substr(0, path.find_last_of('/') + 1).c_str();

	// Load meshes and materials from the OBJ file.
	bool bLoadSuccess = tinyobj::LoadObj(shapes, materials, errorMessage, szFilePath, szMaterialPath);

	if (!bLoadSuccess)
	{
		std::cout << "Error loading OBJ: " + errorMessage << std::endl;
		return;
	}

	// Copy meshes into appropriate buffers...
	m_nMeshChunkCount = static_cast<int>(shapes.size());
	m_meshes = new MeshChunk[m_nMeshChunkCount];

	// -----------------------------------------------------------------------------------------
	// Meshes

	// Array of all vertices of all mesh chunks, for a single mesh VAO.
	std::vector<Vertex> wholeMeshVertices;
	std::vector<unsigned int> wholeMeshIndices;
	wholeMeshVertices.clear();
	wholeMeshIndices.clear();

	// Generate whole mesh buffers...
	glGenBuffers(1, &m_glVBOHandle);
	glGenBuffers(1, &m_glInsHandle);
	glGenBuffers(1, &m_glEBOHandle);
	glGenVertexArrays(1, &m_glVAOHandle);

	for(int i = 0; i < m_nMeshChunkCount; ++i) 
	{
		tinyobj::shape_t& shape = shapes[i];
		MeshChunk& currentChunk = m_meshes[i];

		currentChunk.m_nIndexCount = static_cast<int>(shape.mesh.indices.size());
		currentChunk.m_nMaterialIndex = shape.mesh.material_ids[0];

		// Create mesh buffers...
		glGenBuffers(1, &currentChunk.m_glVBOHandle);
		glGenBuffers(1, &currentChunk.m_glEBOHandle);
		glGenVertexArrays(1, &currentChunk.m_glVAOHandle);

		// Bind VAO.
		glBindVertexArray(currentChunk.m_glVAOHandle);

		// Bind and fill index buffer.
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, currentChunk.m_glEBOHandle);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * shape.mesh.indices.size(), shape.mesh.indices.data(), GL_STATIC_DRAW);

		// Append chunk indices to the whole mesh index array.
		for(int j = 0; j < shape.mesh.indices.size(); ++j) 
		{
			wholeMeshIndices.push_back(shape.mesh.indices[j]);
		}

		std::vector<Vertex> vertices;
		vertices.resize(shape.mesh.positions.size() / 3); // Divide size by 3 to account for the fact that the array is of floats rather than vector structs.

		for(int j = 0; j < vertices.size(); ++j) 
		{
			// Positions, normals etc are stored in float format, in groups. (3 for positions and normals, 2 for tex coords).
			// Multiply the index to jump to the current float group.
			int nIndex = j * 3; // Three floats long for positions and normals.
			int nTexIndex = j * 2; // Two floats long for texture coordinates.

			// Copy positions...
			if (shape.mesh.positions.size())
				vertices[i].m_v4Position = NVZMathLib::Vector4(shape.mesh.positions[nIndex], shape.mesh.positions[nIndex + 1], shape.mesh.positions[nIndex + 2], 1.0f);
			
			// Copy normals...
			if (shape.mesh.normals.size())
				vertices[i].m_v4Normal = NVZMathLib::Vector4(shape.mesh.normals[nIndex], shape.mesh.normals[nIndex + 1], shape.mesh.normals[nIndex + 2], 0.0f);

			// Copy texture coordinates.
			if (shape.mesh.texcoords.size())
				vertices[i].m_v2TexCoords = NVZMathLib::Vector2(shape.mesh.texcoords[nTexIndex], 1 - shape.mesh.texcoords[nTexIndex + 1]);

			// Add the vertex to the whole mesh vertex array.
			wholeMeshVertices.push_back(vertices[i]);
		}

		// Calculate tangents for each vertex...
		CalculateTangents(vertices, shape.mesh.indices);

		// Bind and fill vertex buffer...
		glBindBuffer(GL_ARRAY_BUFFER, currentChunk.m_glVBOHandle);
		glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex) * vertices.size(), vertices.data(), GL_STATIC_DRAW);

		// Vertex attributes...

		// Positions
		glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(float) * 14, (void*)0);
		glEnableVertexAttribArray(0);

		// Normals
		glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(float) * 14, (void*)(sizeof(float) * 4));
		glEnableVertexAttribArray(1);

		// Tangents
		glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(float) * 14, (void*)(sizeof(float) * 8));
		glEnableVertexAttribArray(2);

		// Texture Coordinates
		glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 14, (void*)(sizeof(float) * 12));
		glEnableVertexAttribArray(3);

		// Unbind buffers and VAO
		glBindVertexArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	}

	// Calculate tangents for whole mesh...
	CalculateTangents(wholeMeshVertices, wholeMeshIndices);

	// Fill whole mesh buffers...

	glBindVertexArray(m_glVAOHandle);

	// Bind and fill index buffer...
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_glEBOHandle);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * wholeMeshIndices.size(), wholeMeshIndices.data(), GL_STATIC_DRAW);

	// Bind and fill VBO...
	glBindBuffer(GL_ARRAY_BUFFER, m_glVBOHandle);
	glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex) * wholeMeshVertices.size(), wholeMeshVertices.data(), GL_STATIC_DRAW);

	// Vertex attributes...

	// Positions
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(float) * 14, (void*)0);
	glEnableVertexAttribArray(0);

	// Normals
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(float) * 14, (void*)(sizeof(float) * 4));
	glEnableVertexAttribArray(1);

	// Tangents
	glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(float) * 14, (void*)(sizeof(float) * 8));
	glEnableVertexAttribArray(2);

	// Texture Coordinates
	glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 14, (void*)(sizeof(float) * 12));
	glEnableVertexAttribArray(3);

	// Bind instance buffer to VAO...
	glBindBuffer(GL_ARRAY_BUFFER, m_glInsHandle);
	glBufferData(GL_ARRAY_BUFFER, sizeof(Instance) * MAX_INSTANCE_COUNT, 0, GL_STATIC_DRAW);

	// Instance Attributes...

	// Color
	glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, sizeof(Instance), (void*)0);
	glEnableVertexAttribArray(4);

	glVertexAttribDivisor(4, 1);

	// Model matrix
	glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, sizeof(Instance), (void*)(sizeof(float) * 4));
	glEnableVertexAttribArray(5);

	glVertexAttribPointer(6, 4, GL_FLOAT, GL_FALSE, sizeof(Instance), (void*)(sizeof(float) * 8));
	glEnableVertexAttribArray(6);

	glVertexAttribPointer(7, 4, GL_FLOAT, GL_FALSE, sizeof(Instance), (void*)(sizeof(float) * 12));
	glEnableVertexAttribArray(7);

	glVertexAttribPointer(8, 4, GL_FLOAT, GL_FALSE, sizeof(Instance), (void*)(sizeof(float) * 16));
	glEnableVertexAttribArray(8);

	glVertexAttribDivisor(5, 1);
	glVertexAttribDivisor(6, 1);
	glVertexAttribDivisor(7, 1);
	glVertexAttribDivisor(8, 1);

	// Normal matrix
	glVertexAttribPointer(9, 3, GL_FLOAT, GL_FALSE, sizeof(Instance), (void*)(sizeof(float) * 20));
	glEnableVertexAttribArray(9);
	
	glVertexAttribPointer(10, 3, GL_FLOAT, GL_FALSE, sizeof(Instance), (void*)(sizeof(float) * 23));
	glEnableVertexAttribArray(10);
	
	glVertexAttribPointer(11, 3, GL_FLOAT, GL_FALSE, sizeof(Instance), (void*)(sizeof(float) * 26));
	glEnableVertexAttribArray(11);
	
	glVertexAttribDivisor(9, 1);
	glVertexAttribDivisor(10, 1);
	glVertexAttribDivisor(11, 1);

	// Unbind buffers and VAO
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	m_nWholeVertexCount = static_cast<unsigned int>(wholeMeshVertices.size());
	m_nWholeIndexCount = static_cast<unsigned int>(wholeMeshIndices.size());

	// -----------------------------------------------------------------------------------------
	// Materials

	m_materials = new Material*[materials.size()];

	for (int i = 0; i < materials.size(); ++i)
	{
		tinyobj::material_t& currentMaterial = materials[i];
		Material*& currentFinalMat = m_materials[i];

		// Allocate material memory.
		currentFinalMat = new Material(nullptr);

		// Set material properties...
		currentFinalMat->SetAmbientColor(currentMaterial.ambient[0], currentMaterial.ambient[1], currentMaterial.ambient[1]);
		currentFinalMat->SetDiffuseColor(currentMaterial.diffuse[0], currentMaterial.diffuse[1], currentMaterial.diffuse[1]);
		currentFinalMat->SetSpecularColor(currentMaterial.specular[0], currentMaterial.specular[1], currentMaterial.specular[1]);
		currentFinalMat->SetEmissionColor(currentMaterial.emission[0], currentMaterial.emission[1], currentMaterial.emission[1]);

		currentFinalMat->SetSpecularShininess(currentMaterial.shininess);
		currentFinalMat->SetAlpha(currentMaterial.dissolve);

		// Add maps in order...

		// Diffuse
		if (!currentMaterial.diffuse_texname.empty() && (textureFlags & TEXTURE_MAP_DIFFUSE)) 
		{
			Texture* newMap = new Texture((szMaterialPath + currentMaterial.diffuse_texname).c_str());
			currentFinalMat->AddMap(newMap);

			m_textureMaps.Push(newMap);
		}

		// Normal
		if (!currentMaterial.bump_texname.empty() && (textureFlags & TEXTURE_MAP_NORMAL))
		{
			Texture* newMap = new Texture((szMaterialPath + currentMaterial.bump_texname).c_str());
			currentFinalMat->AddMap(newMap);

			m_textureMaps.Push(newMap);
		}

		// Specular
		if (!currentMaterial.specular_texname.empty() && (textureFlags & TEXTURE_MAP_SPECULAR))
		{
			Texture* newMap = new Texture((szMaterialPath + currentMaterial.specular_texname).c_str());
			currentFinalMat->AddMap(newMap);

			m_textureMaps.Push(newMap);
		}

		// Specular
		if (!currentMaterial.specular_highlight_texname.empty() && (textureFlags & TEXTURE_MAP_SPECULARHIGHLIGHT))
		{
			Texture* newMap = new Texture((szMaterialPath + currentMaterial.specular_highlight_texname).c_str());
			currentFinalMat->AddMap(newMap);

			m_textureMaps.Push(newMap);
		}

		// Specular Highlight
		if (!currentMaterial.displacement_texname.empty() && (textureFlags & TEXTURE_MAP_DISPLACEMENT))
		{
			Texture* newMap = new Texture((szMaterialPath + currentMaterial.displacement_texname).c_str());
			currentFinalMat->AddMap(newMap);

			m_textureMaps.Push(newMap);
		}

		// Alpha
		if (!currentMaterial.alpha_texname.empty() && (textureFlags & TEXTURE_MAP_ALPHA))
		{
			Texture* newMap = new Texture((szMaterialPath + currentMaterial.alpha_texname).c_str());
			currentFinalMat->AddMap(newMap);

			m_textureMaps.Push(newMap);
		}

		// Ambient
		if (!currentMaterial.ambient_texname.empty() && (textureFlags & TEXTURE_MAP_AMBIENT))
		{
			Texture* newMap = new Texture((szMaterialPath + currentMaterial.ambient_texname).c_str());
			currentFinalMat->AddMap(newMap);

			m_textureMaps.Push(newMap);
		}
	}

	// -----------------------------------------------------------------------------------------
}

void Mesh::SetShader(Shader* shader, int nMaterialIndex) 
{
	m_materials[nMaterialIndex % m_nMaterialCount]->SetShader(shader);
}

void Mesh::Bind() 
{
	glBindVertexArray(m_glVAOHandle);
	glBindBuffer(GL_ARRAY_BUFFER, m_glVBOHandle);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_glEBOHandle);
	glBindBuffer(GL_ARRAY_BUFFER, m_glInsHandle);
}

unsigned int Mesh::VBOHandle() 
{
	return m_glVBOHandle;
}

unsigned int Mesh::IndexBufferHandle() 
{
	return m_glEBOHandle;
}

unsigned int Mesh::VAOHandle() 
{
	return m_glVAOHandle;
}

unsigned int Mesh::InstanceHandle() 
{
	return m_glInsHandle;
}

unsigned int Mesh::VertexCount() 
{
	return m_nWholeVertexCount;
}

unsigned int Mesh::IndexCount() 
{
	return m_nWholeIndexCount;
}

void Mesh::CalculateTangents(std::vector<Vertex>& vertices, std::vector<unsigned int>& indices) 
{
	// Lengyel, Eric. “Computing Tangent Space Basis Vectors for an Arbitrary Mesh”. Terathon Software, 2001. http://terathon.com/code/tangent.html

	unsigned int vertexCount = (unsigned int)vertices.size();
	glm::vec4* tan1 = new glm::vec4[vertexCount * 2];
	glm::vec4* tan2 = tan1 + vertexCount;
	memset(tan1, 0, vertexCount * sizeof(glm::vec4) * 2);

	unsigned int indexCount = (unsigned int)indices.size();
	for (unsigned int a = 0; a < indexCount; a += 3) {
		long i1 = indices[a];
		long i2 = indices[a + 1];
		long i3 = indices[a + 2];

		const NVZMathLib::Vector4& v1 = vertices[i1].m_v4Position;
		const NVZMathLib::Vector4& v2 = vertices[i2].m_v4Position;
		const NVZMathLib::Vector4& v3 = vertices[i3].m_v4Position;

		const NVZMathLib::Vector2& w1 = vertices[i1].m_v2TexCoords;
		const NVZMathLib::Vector2& w2 = vertices[i2].m_v2TexCoords;
		const NVZMathLib::Vector2& w3 = vertices[i3].m_v2TexCoords;

		float x1 = v2.x - v1.x;
		float x2 = v3.x - v1.x;
		float y1 = v2.y - v1.y;
		float y2 = v3.y - v1.y;
		float z1 = v2.z - v1.z;
		float z2 = v3.z - v1.z;

		float s1 = w2.x - w1.x;
		float s2 = w3.x - w1.x;
		float t1 = w2.y - w1.y;
		float t2 = w3.y - w1.y;

		float r = 1.0F / (s1 * t2 - s2 * t1);
		glm::vec4 sdir((t2 * x1 - t1 * x2) * r, (t2 * y1 - t1 * y2) * r,
			(t2 * z1 - t1 * z2) * r, 0);
		glm::vec4 tdir((s1 * x2 - s2 * x1) * r, (s1 * y2 - s2 * y1) * r,
			(s1 * z2 - s2 * z1) * r, 0);

		tan1[i1] += sdir;
		tan1[i2] += sdir;
		tan1[i3] += sdir;

		tan2[i1] += tdir;
		tan2[i2] += tdir;
		tan2[i3] += tdir;
	}

	for (unsigned int a = 0; a < vertexCount; a++) 
	{
		const glm::vec3& n = glm::vec3(vertices[a].m_v4Normal.x, vertices[a].m_v4Normal.y, vertices[a].m_v4Normal.z);
		const glm::vec3& t = glm::vec3(tan1[a]);

		// Gram-Schmidt orthogonalize
		glm::vec4 orthTangent = glm::vec4(glm::normalize(t - n * glm::dot(n, t)), 0);
		vertices[a].m_v4Tangent = NVZMathLib::Vector4(orthTangent.x, orthTangent.y, orthTangent.z, orthTangent.w);

		// Calculate handedness (direction of bitangent)
		vertices[a].m_v4Tangent.w = (glm::dot(glm::cross(glm::vec3(n), glm::vec3(t)), glm::vec3(tan2[a])) < 0.0F) ? 1.0F : -1.0F;
	}

	delete[] tan1;
}