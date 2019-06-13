#pragma once
#include <vector>
#include <string>

struct CUfunc_st;
struct CUmod_st;
struct _nvrtcProgram;

namespace GPL 
{
#define STANDARD_HEADER_COUNT 4

	typedef CUfunc_st* CUfunction;
	typedef CUmod_st* CUmodule;
	typedef _nvrtcProgram* nvrtcProgram;

	class GPL_BehaviourProgram 
	{
	public:

		GPL_BehaviourProgram(const char* sourcePath);

		~GPL_BehaviourProgram();

		// Compile particle behaviour GPU program...
		void CompileProgram();

		// Attach a header file for use in the particle behaviour GPU program.
		void AttachHeader(const char* headerPath);

		// Get the handle for the particle behaviour GPU program.
		const nvrtcProgram& GetProgram();

		const CUfunction& InitFunction();

		const CUfunction& UpdateFunction();

	private:

		typedef unsigned long long devicePtr;

		std::string LoadSource(const char* path);

		// CUDA handles
		nvrtcProgram m_program;
		CUmodule m_module;
		CUfunction m_initKernel;
		CUfunction m_updateKernel;

		// Kernel code
		static std::string m_standardHeaderSources[STANDARD_HEADER_COUNT];

		std::vector<char*> m_attachedSources;
		std::vector<char*> m_attachedNames;
		std::string m_kernelSource;

		size_t m_ptxSize;
	};
}
