#pragma once
#include "Camera.h"

#include "GPL.h"
#include "GPL_ParticleSystem.h"
#include "GPL_Behaviour.h"
#include "Vector3.h"

#define STAR_FIELD_COUNT 100

struct GLFWwindow;

class Input;
class Renderer;

enum EParticleState 
{
	PARTICLE_STARS,
	PARTICLE_SPHERE_DISSOLVE,
	PARTICLE_BLACK_HOLE
};

class Application
{
public:

	Application();

	~Application();

	int Init();

	void Run();

private:

	// GLFW Callbacks
	static void ErrorCallBack(int error, const char* desc);
	static void KeyCallBack(GLFWwindow* window, int key, int scancode, int action, int mods);
	static void MouseButtonCallBack(GLFWwindow* window, int button, int action, int mods);
	static void CursorPosCallBack(GLFWwindow* window, double dXPos, double dYPos);
	static void MouseScrollCallBack(GLFWwindow* window, double dXOffset, double dYOffset);

	GLFWwindow* m_window;
	static Renderer* m_renderer;
	static Input* m_input;
	bool m_bGLFWInit;

	Camera m_camera;

	// State
	EParticleState m_particleState;

	// Star effect.
	GPL::GPL_ParticleSystem* m_starParticles[STAR_FIELD_COUNT];
	GPL::GPL_BehaviourProgram* m_starBehaviour;

	// Dissolve effect.
	GPL::GPL_ParticleSystem* m_sphereDissolveEffect;
	GPL::GPL_BehaviourProgram* m_sphereDissolveBehaviour;

	// Black hole effect.
	GPL::GPL_GlobalP3DData m_blackHoleData;
	GPL::GPL_ParticleSystem* m_blackHoleEffect;
	GPL::GPL_BehaviourProgram* m_blackHoleBehaviour;

	NVZMathLib::Vector3 m_v3ParticleTarget;
};

