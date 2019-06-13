#pragma once
#include <string>
#include "Vector4.h"
#include "Matrix4.h"

class Shader
{
public:

	Shader(const char* szVertShaderPath, const char* szFragShaderPath);

	~Shader();

	/*
	Description: Use/Bind the shader for use with the next GPU draw call.
	*/
	void Use();

	/*
	Description: Reset shader program binding to default.
	*/
	static void ResetBinding();

	/*
	Description: Get the OpenGL handle of this shader program.
	Return Type: unsigned int
	*/
	unsigned int GetHandle();

	// Set uniform value in shader.
	void SetUniformInt(const char* szUniformName, int nValue);

	// Set uniform value in shader.
	void SetUniformInt(unsigned int glUniformLocation, int nValue);

	// Set uniform value in shader.
	void SetUniformFloat(const char* szUniformName, float fValue);

	// Set uniform value in shader.
	void SetUniformFloat(unsigned int glUniformLocation, float fValue);

	// Set uniform value in shader.
	void SetUniformVec2(const char* szUniformName, NVZMathLib::Vector2 v2Value);

	// Set uniform value in shader.
	void SetUniformVec2(unsigned int glUniformLocation, NVZMathLib::Vector2 v2Value);

	// Set uniform value in shader.
	void SetUniformVec3(const char* szUniformName, NVZMathLib::Vector3 v3Value);

	// Set uniform value in shader.
	void SetUniformVec3(unsigned int glUniformLocation, NVZMathLib::Vector3 v3Value);

	// Set uniform value in shader.
	void SetUniformVec4(const char* szUniformName, NVZMathLib::Vector4 v4Value);

	// Set uniform value in shader.
	void SetUniformVec4(unsigned int glUniformLocation, NVZMathLib::Vector4 v4Value);

	// Set uniform value in shader.
	void SetUniformMat4(const char* szUniformName, NVZMathLib::Matrix4 value);

private:

	// Loads the shader contents from the specified shader paths.
	void Load();

	const char* m_szVertShaderPath;
	const char* m_szFragShaderPath;

	std::string m_szVertShaderContents;
	std::string m_szFragShaderContents;

	unsigned int m_glHandle;
};

