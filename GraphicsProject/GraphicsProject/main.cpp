#include "Application.h"

int main() 
{
	Application* application = new Application();

	// Initialize and quit if the code is not zero.
	int code = application->Init();
	if (code)
		return code;

	// Run application.
	application->Run();

	delete application;

	return 0;
}