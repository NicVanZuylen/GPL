#include "Shader.h"
#include <iostream>
#include <fstream>
#include <sstream>

#include "glad/glad.h"

#include "Renderer.h"

Shader::Shader(const char* szVertShaderPath, const char* szFragShaderPath)
{
	m_glHandle = 0;

	m_szVertShaderPath = szVertShaderPath;
	m_szFragShaderPath = szFragShaderPath;

	// Load shader contents from shader files.
	Load();

	// Create shader program.
	m_glHandle = glCreateProgram();

	const char* szVShaderCont = m_szVertShaderContents.c_str();
	const char* szFShaderCont = m_szFragShaderContents.c_str();

	// Shader compilation information.
	int nCompilationSuccessful = 0;
	char compilerOutput[512];

	// -----------------------------------------------------------------------------------------
	// Vertex Shader

	unsigned int glVertShaderHandle = 0;
	glVertShaderHandle = glCreateShader(GL_VERTEX_SHADER);

	// Set shader source code.
	glShaderSource(glVertShaderHandle, 1, &szVShaderCont, nullptr);

	// Compile vertex shader.
	glCompileShader(glVertShaderHandle);

	// Get shader compilation results.
	glGetShaderiv(glVertShaderHandle, GL_COMPILE_STATUS, &nCompilationSuccessful);

	// If the compilation failed get the error output and print it to the console.
	if(!nCompilationSuccessful) 
	{
		glGetShaderInfoLog(glVertShaderHandle, 512, nullptr, compilerOutput);

		std::cout << "Shader Error: Vertex Shader compilation failed:\n" << compilerOutput << std::endl;
	}

	// -----------------------------------------------------------------------------------------
	// Fragment Shader

	unsigned int glFragShaderHandle = 0;
	glFragShaderHandle = glCreateShader(GL_FRAGMENT_SHADER);

	// Set shader source code.
	glShaderSource(glFragShaderHandle, 1, &szFShaderCont, nullptr);

	// Compile vertex shader.
	glCompileShader(glFragShaderHandle);

	// Get shader compilation results.
	glGetShaderiv(glFragShaderHandle, GL_COMPILE_STATUS, &nCompilationSuccessful);

	// If the compilation failed get the error output and print it to the console.
	if (!nCompilationSuccessful)
	{
		glGetShaderInfoLog(glFragShaderHandle, 512, nullptr, compilerOutput);

		std::cout << "Shader Error: Fragment Shader compilation failed:\n" << compilerOutput << std::endl;
	}

	// -----------------------------------------------------------------------------------------
	// Shader program finalization

	// Attach shaders to the shader program.
	glAttachShader(m_glHandle, glVertShaderHandle);
	glAttachShader(m_glHandle, glFragShaderHandle);


	// Link shader program.
	glLinkProgram(m_glHandle);

	// Link error logging...
	glGetProgramiv(m_glHandle, GL_LINK_STATUS, &nCompilationSuccessful);

	// Log any linking errors...
	if (!nCompilationSuccessful)
	{
		glGetProgramInfoLog(m_glHandle, 512, nullptr, compilerOutput);

		std::cout << "Shader Program Link Error: Program linking failed:\n" << compilerOutput << std::endl;
	}

	// Delete independent shaders...
	glDeleteShader(glVertShaderHandle);
	glDeleteShader(glFragShaderHandle);

	// Uniform block bindings...
	unsigned int glMatrixUBOhandle = glGetUniformBlockIndex(m_glHandle, "GlobalMatrices");

	glUniformBlockBinding(m_glHandle, glMatrixUBOhandle, 0);

	// -----------------------------------------------------------------------------------------
	// Sampler uniforms.
	Use();

	for (int i = 0; i < 16; ++i)
	{
		// Get the name of the current sampler block...
		std::string szSamplerName = "textureMaps[" + std::to_string(i) + "]";

		int nSamplerLocation = glGetUniformLocation(m_glHandle, szSamplerName.c_str());

		// Assign sampler location...
		glUniform1i(nSamplerLocation, i);
	}

	glUseProgram(0);
	// -----------------------------------------------------------------------------------------
}

Shader::~Shader()
{
	glDeleteProgram(m_glHandle);
}

void Shader::Use() 
{
	glUseProgram(m_glHandle);
}

void Shader::ResetBinding() 
{
	glUseProgram(0);
}

unsigned int Shader::GetHandle() 
{
	return m_glHandle;
}

void Shader::SetUniformInt(const char* szUniformName, int nValue)
{
	unsigned int glLocation = glGetUniformLocation(m_glHandle, szUniformName);
	glUniform1iv(glLocation, 1, &nValue);
}

void Shader::SetUniformInt(unsigned int glUniformLocation, int nValue)
{
	glUniform1i(glUniformLocation, nValue);
}

void Shader::SetUniformFloat(const char* szUniformName, float fValue)
{
	unsigned int glLocation = glGetUniformLocation(m_glHandle, szUniformName);
	glUniform1f(glLocation, fValue);
}

void Shader::SetUniformFloat(unsigned int glUniformLocation, float fValue)
{
	glUniform1f(glUniformLocation, fValue);
}

void Shader::SetUniformVec2(const char* szUniformName, NVZMathLib::Vector2 v2Value)
{
	unsigned int glLocation = glGetUniformLocation(m_glHandle, szUniformName);
	glUniform2fv(glLocation, 1, (float*)v2Value);
}

void Shader::SetUniformVec2(unsigned int glUniformLocation, NVZMathLib::Vector2 v2Value)
{
	glUniform2fv(glUniformLocation, 1, (float*)v2Value);
}

void Shader::SetUniformVec3(const char* szUniformName, NVZMathLib::Vector3 v3Value)
{
	unsigned int glLocation = glGetUniformLocation(m_glHandle, szUniformName);
	glUniform3fv(glLocation, 1, (float*)v3Value);
}

void Shader::SetUniformVec3(unsigned int glUniformLocation, NVZMathLib::Vector3 v3Value)
{
	glUniform3fv(glUniformLocation, 1, (float*)v3Value);
}

void Shader::SetUniformVec4(const char* szUniformName, NVZMathLib::Vector4 v4Value)
{
	unsigned int glLocation = glGetUniformLocation(m_glHandle, szUniformName);
	glUniform4fv(glLocation, 1, (float*)v4Value);
}

void Shader::SetUniformVec4(unsigned int glUniformLocation, NVZMathLib::Vector4 v4Value)
{
	glUniform4fv(glUniformLocation, 1, (float*)v4Value);
}

void Shader::SetUniformMat4(const char* szUniformName, NVZMathLib::Matrix4 value)
{
	unsigned int glLocation = glGetUniformLocation(m_glHandle, szUniformName);
	
	glUniformMatrix4fv(glLocation, 1, GL_FALSE, (float*)value);
}

void Shader::Load() 
{
	// Input file stream
	std::ifstream vertexShaderFile;
	std::ifstream fragmentShaderFile;

	// Exceptions
	vertexShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	fragmentShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);

	try
	{
		// Open files...
		vertexShaderFile.open(m_szVertShaderPath);
		fragmentShaderFile.open(m_szFragShaderPath);

		// Read file buffers...
		std::stringstream vertexShaderData;
		std::stringstream fragmentShaderData;

		vertexShaderData << vertexShaderFile.rdbuf();
		fragmentShaderData << fragmentShaderFile.rdbuf();

		// Close files.
		vertexShaderFile.close();
		fragmentShaderFile.close();

		// Convert data to strings.
		m_szVertShaderContents = vertexShaderData.str();
		m_szFragShaderContents = fragmentShaderData.str();
	}
	catch (std::fstream::failure e)
	{
		std::cout << "Failed to load shaders at:\n"
			<< m_szVertShaderPath
			<< "\n and:\n"
			<< m_szFragShaderPath
			<< "\n Error: "
			<< e.what()
			<< std::endl;
	}
}
