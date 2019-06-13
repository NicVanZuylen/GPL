#include "GPL_Behaviour.h"
#include <fstream>
#include <sstream>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

using namespace GPL;

std::string GPL_BehaviourProgram::m_standardHeaderSources[STANDARD_HEADER_COUNT];

GPL_BehaviourProgram::GPL_BehaviourProgram(const char* sourcePath) 
{
	// Load file...
	m_kernelSource = LoadSource(sourcePath);
}

void GPL_BehaviourProgram::CompileProgram() 
{
	// Load GPL headers if they are not already loaded.
	if (m_standardHeaderSources[0].empty())
	{
		m_standardHeaderSources[0] = LoadSource("GPL/GPL_RT_RandStates.cuh");
		m_standardHeaderSources[1] = LoadSource("GPL/GPL_RT_ParticleBehaviours.cuh");
		m_standardHeaderSources[2] = LoadSource("GPL/GPL_RT_Vectors.cuh");
		m_standardHeaderSources[3] = LoadSource("GPL/thirdparty/rngpu.hpp");
	}

	int finalSize = STANDARD_HEADER_COUNT + (int)m_attachedNames.size();

	const char** headerSources = new const char*[finalSize]; // Temporary.

	// Add standard headers...
	headerSources[0] = m_standardHeaderSources[0].c_str();
	headerSources[1] = m_standardHeaderSources[1].c_str();
	headerSources[2] = m_standardHeaderSources[2].c_str();
	headerSources[3] = m_standardHeaderSources[3].c_str();

	// Add attached headers...
	for(int i = 0; i < m_attachedSources.size(); ++i) 
	{
		headerSources[i + STANDARD_HEADER_COUNT] = m_attachedSources[i];
	}

	const char** includeNames = new const char*[finalSize]; // Temporary.

	// Add standard header names...
	includeNames[0] = "GPL_RT_RandStates.cuh";
	includeNames[1] = "GPL_RT_ParticleBehaviours.cuh";
	includeNames[2] = "GPL_RT_Vectors.cuh";
	includeNames[3] = "rngpu.hpp";
	
	// Add attached header names...
	for (int i = 0; i < m_attachedNames.size(); ++i)
	{
		includeNames[i + STANDARD_HEADER_COUNT] = m_attachedNames[i];
	}

	// Create program...
	nvrtcResult error = nvrtcCreateProgram(&m_program, m_kernelSource.c_str(), "GPL program", finalSize, headerSources, includeNames);

	if (error)
	{
		std::cout << nvrtcGetErrorString(error) << "\n";
	}

	error = nvrtcCompileProgram(m_program, 0, 0);

	if (error)
	{
		std::cout << nvrtcGetErrorString(error) << "\n";

		// Get log size...
		size_t logSize;
		nvrtcGetProgramLogSize(m_program, &logSize);

		char* log = new char[logSize];

		// Get log data.
		nvrtcGetProgramLog(m_program, log);

		// Print log.
		std::cout << log << "\n";

		delete[] headerSources;
		delete[] includeNames;

		delete[] log;

		return;
	}

	delete[] headerSources;
	delete[] includeNames;

	// Get program PTX:

	// Get size.
	nvrtcGetPTXSize(m_program, &m_ptxSize);

	// Allocate memory.
	char* kernelPTX = new char[m_ptxSize];

	// Fetch PTX data.
	nvrtcGetPTX(m_program, kernelPTX);

	// Create module from PTX.
	CUresult result = cuModuleLoadDataEx(&m_module, kernelPTX, 0, 0, 0);

	// Get device functions.
	result = cuModuleGetFunction(&m_initKernel, m_module, "InitKernel");
	result = cuModuleGetFunction(&m_updateKernel, m_module, "UpdateKernel");
}

GPL_BehaviourProgram::~GPL_BehaviourProgram() 
{
	nvrtcDestroyProgram(&m_program);

	// Unload PTX module.
	cuModuleUnload(m_module);

	// Delete string data...

	for (int i = 0; i < m_attachedNames.size(); ++i)
		delete[] m_attachedNames[i];

	for (int i = 0; i < m_attachedSources.size(); ++i)
		delete[] m_attachedSources[i];
}

void GPL_BehaviourProgram::AttachHeader(const char* headerPath) 
{
	std::string strPath = headerPath;
	size_t nameLoc = strPath.find_last_of('/'); // Gets the header file name location in the string.
	std::string name = strPath.substr(nameLoc);

	// Copy name to allocated C string.
	char* nameCString = new char[name.size()];
	strcpy_s(nameCString, name.size(), name.c_str());

	m_attachedNames.push_back(nameCString);

	// Copy source code to allocated C string.
	std::string source = LoadSource(headerPath);
	char* sourceCString = new char[source.size()];
	strcpy_s(sourceCString, source.size(), source.c_str());

	m_attachedSources.push_back(sourceCString);
}

std::string GPL_BehaviourProgram::LoadSource(const char* path) 
{
	std::fstream sourceInputStream;

	// Exceptions...
	sourceInputStream.exceptions(std::ifstream::failbit | std::ifstream::badbit);

	try
	{
		// Open file...
		sourceInputStream.open(path);

		std::stringstream fileContents;

		// Read file contents...
		fileContents << sourceInputStream.rdbuf();

		// Close file.
		sourceInputStream.close();

		return fileContents.str();
	}
	catch (std::fstream::failure e)
	{
		std::cout << "GPL: Failed to load file source at: " << path << " Error: " << e.what() << "\n";
	}

	return "GPL_LOAD_FAIL";
}

const nvrtcProgram& GPL_BehaviourProgram::GetProgram() 
{
	return m_program;
}

const CUfunction& GPL_BehaviourProgram::InitFunction() 
{
	return m_initKernel;
}

const CUfunction& GPL_BehaviourProgram::UpdateFunction()
{
	return m_updateKernel;
}