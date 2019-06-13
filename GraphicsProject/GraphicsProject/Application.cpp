#include "glfw3.h"

#include "Application.h"
#include "Renderer.h"
#include "Input.h"

#include <iostream>
#include <chrono>

#include "Shader.h"
#include "Texture.h"
#include "Material.h"
#include "Mesh.h"
#include "Batch.h"
#include "StaticMeshRenderer.h"
#include "MeshRenderer.h"
#include "FrameBuffer.h"
#include "Matrix4.h"
#include "NewMatrix4.h"
#include "glm.hpp"
#include "glm\include\gtc\quaternion.hpp"
#include "glm/include/ext.hpp"

#include "RenderSingle.h"

#include "GPL_VertexAttributes.h"

#define MOUSE_SENSITIVITY 0.1f
#define CAMERA_MOVE_SPEED 5.0f
#define BLOOM_PASS_COUNT 3

using namespace NVZMathLib;

using namespace GPL;

Input* Application::m_input = nullptr;
Renderer* Application::m_renderer = nullptr;

Application::Application()
{

}

Application::~Application()
{
	if (m_bGLFWInit)
		glfwTerminate();
	else
		return;

	// Destroy renderer.
	delete m_renderer;

	// Destroy window.
	glfwDestroyWindow(m_window);

	GPL::GPL_Main::Quit();

	// Destroy input.
	Input::Destroy();
}

int Application::Init() 
{
	m_bGLFWInit = false;

	if (!glfwInit())
		return -1;

	m_bGLFWInit = true;

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// Create window.
	m_window = glfwCreateWindow(1280, 720, "GPL Demo", 0, 0);
	glfwMaximizeWindow(m_window);

	GLFWmonitor* mainMonitor = glfwGetPrimaryMonitor();

	// Set key callback...
	glfwSetKeyCallback(m_window, &KeyCallBack);

	// Set mouse callbacks...
	glfwSetMouseButtonCallback(m_window, &MouseButtonCallBack);
	glfwSetCursorPosCallback(m_window, &CursorPosCallBack);
	glfwSetScrollCallback(m_window, &MouseScrollCallBack);

	// Create renderer.
	m_renderer = new Renderer(m_window);

	GPL_Main::Init();

	m_starBehaviour = new GPL_BehaviourProgram("GPL/gpl_example1.cu");
	m_starBehaviour->CompileProgram();

	m_sphereDissolveBehaviour = new GPL_BehaviourProgram("GPL/gpl_example2.cu");
	m_sphereDissolveBehaviour->CompileProgram();

	m_blackHoleBehaviour = new GPL_BehaviourProgram("GPL/gpl_example3.cu");
	m_blackHoleBehaviour->CompileProgram();

	GPL_ParticleSystem::CreateStream();

	// Star effect initialization.
	for(int i = 0; i < STAR_FIELD_COUNT; ++i)
	{
		m_starParticles[i] = new GPL_ParticleSystem(m_starBehaviour, 1000);
		GPL_BaseData baseData;
		GPL_GlobalP3DData globalData;
	
		baseData.m_burstInterval = 0.01f;
		baseData.m_burstCount = 5;
	
		globalData.m_origin = { i * 8.0f, 5.0f, 0.0f };
		globalData.m_startColor = { 0.0f, 1.0f, 1.0f };
		globalData.m_spawnScale = 0.01f;
	
		m_starParticles[i]->SetBaseData(baseData);
		m_starParticles[i]->SetGlobalData<GPL_GlobalP3DData>(&globalData);
	
		m_starParticles[i]->Initialize(i * 321337);
	}

	// Dissolve effect initialization.
	m_sphereDissolveEffect = new GPL_ParticleSystem(m_sphereDissolveBehaviour, 10000);

	{
		GPL_BaseData baseData;
		GPL_GlobalP3DData globalData;

		baseData.m_burstInterval = 0.01f;
		baseData.m_burstCount = 1000;

		globalData.m_origin = { 0.0f, 5.0f, 0.0f };
		globalData.m_startColor = { 0.0f, 1.0f, 1.0f };
		globalData.m_spawnScale = 0.01f;

		m_sphereDissolveEffect->SetBaseData(baseData);
		m_sphereDissolveEffect->SetGlobalData<GPL_GlobalP3DData>(&globalData);

		m_sphereDissolveEffect->Initialize(321337);
	}

	m_blackHoleEffect = new GPL_ParticleSystem(m_blackHoleBehaviour, 100000);

	{
		GPL_BaseData baseData;
		GPL_GlobalP3DData& globalData = m_blackHoleData;

		baseData.m_burstInterval = 0.01f;
		baseData.m_burstCount = 100;

		globalData.m_origin = { 0.0f, 5.0f, 0.0f };
		globalData.m_target = { 0.0f, 5.0f, 0.0f };
		globalData.m_startColor = { 0.0f, 1.0f, 1.0f };
		globalData.m_spawnScale = 0.01f;

		m_v3ParticleTarget = { globalData.m_target.x, globalData.m_target.y, globalData.m_target.z };

		m_blackHoleEffect->SetBaseData(baseData);
		m_blackHoleEffect->SetGlobalData<GPL_GlobalP3DData>(&globalData);

		m_blackHoleEffect->Initialize(321337);
	}

	m_particleState = EParticleState::PARTICLE_STARS;
	
	// Initialize input.
	Input::Create();
	m_input = Input::GetInstance();

	// Initialize camera.
	m_camera = Camera({ 0.0f, 2.5f, 5.0f }, { 0.0f, 0.0f, 0.0f }, 0.05f, 10.0f);

	return 0;
}

void Application::Run() 
{
	// Fullscreen quad shaders.
	Shader* quadShader = new Shader("Shaders/quad/standard_fsquad.vs", "Shaders/quad/standard_fsquad.fs");
	Shader* brightShader = new Shader("Shaders/quad/standard_fsquad.vs", "Shaders/quad/standard_bright.fs");
	Shader* bloomHDRShader = new Shader("Shaders/quad/standard_fsquad.vs", "Shaders/quad/standard_bloom_hdr.fs");
	Shader* directionalLightSdr = new Shader("Shaders/quad/standard_fsquad.vs", "Shaders/light/deferred_directional_light_pbr.fs");

	Shader* gaussianHrzntl = new Shader("Shaders/quad/standard_fsquad.vs", "Shaders/quad/standard_gaussian_h.fs");
	Shader* gaussianVert = new Shader("Shaders/quad/standard_fsquad.vs", "Shaders/quad/standard_gaussian_v.fs");

	gaussianHrzntl->Use();
	gaussianHrzntl->SetUniformFloat("power", 0.003f);

	gaussianVert->Use();
	gaussianVert->SetUniformFloat("power", 0.003f);

	bloomHDRShader->Use();
	bloomHDRShader->SetUniformFloat("exposure", 1.0f);

	Shader::ResetBinding();

	// Scene object shaders.
	Shader* planeShader = new Shader("Shaders/plain.vs", "Shaders/plain.fs");

	Shader* particleShader = new Shader("Shaders/particle.vs", "Shaders/particle.fs");

	Shader* lightShader = new Shader("Shaders/light/deferred_point_light_pbr.vs", "Shaders/light/deferred_point_light_pbr.fs");

	Mesh* planeMesh = new Mesh("Assets/Primitives/plane.obj");
	Mesh* sphereMesh = new Mesh("Assets/Primitives/sphere.obj");

	// Particle meshes.
	GPL_VertexAttributes attributes;
	attributes.AddAttribute(GPL_FLOAT4, 0); // Position
	attributes.AddAttribute(GPL_FLOAT4, 0); // Normal
	attributes.AddAttribute(GPL_FLOAT4, 0); // Tangent
	attributes.AddAttribute(GPL_FLOAT2, 0); // Tex Coords

	for(int i = 0; i < STAR_FIELD_COUNT; ++i)
	{
		m_starParticles[i]->SetMesh(sphereMesh->VBOHandle(), sphereMesh->IndexBufferHandle(), sphereMesh->IndexCount(), attributes);
	}

	m_sphereDissolveEffect->SetMesh(sphereMesh->VBOHandle(), sphereMesh->IndexBufferHandle(), sphereMesh->IndexCount(), attributes);

	m_blackHoleEffect->SetMesh(sphereMesh->VBOHandle(), sphereMesh->IndexBufferHandle(), sphereMesh->IndexCount(), attributes);

	Material* floorMat = new Material(planeShader);

	StaticMeshRenderer planeStaticMesh(floorMat);
	planeStaticMesh.PushMesh(planeMesh, glm::value_ptr(glm::mat4()));

	planeStaticMesh.FinalizeBuffers();

	Batch* planeBatch = new Batch(planeMesh, floorMat);

	m_renderer->AddBatch(planeBatch);

	glm::tquat<float> modelQuat;
	glm::mat4 modelMatrix; // For the spear.
	glm::mat4 floorModelMatrix;

	Framebuffer* gBuffer = new Framebuffer(m_renderer->WindowWidth(), m_renderer->WindowHeight());
	gBuffer->AddBufferColorAttachment(BUFFER_FLOAT_RGBA16); // Diffuse buffer.
	gBuffer->AddBufferColorAttachment(BUFFER_FLOAT_RGBA16); // Position buffer.
	gBuffer->AddBufferColorAttachment(BUFFER_FLOAT_RGBA16); // Normal buffer.
	gBuffer->AddBufferColorAttachment(BUFFER_FLOAT_RGBA16); // Specular buffer.
	gBuffer->AddBufferColorAttachment(BUFFER_RGB); // Roughness, spec strength, reflection coefficent
	gBuffer->AddBufferColorAttachment(BUFFER_RGB); // Emission
	gBuffer->AddDepthAttachment();

	// Color buffer for HDR.
	Framebuffer* brightColorBuffer = new Framebuffer(m_renderer->WindowWidth(), m_renderer->WindowHeight());
	brightColorBuffer->AddBufferColorAttachment(BUFFER_RGB); // Bright color.

	Framebuffer* bloomBuffer = new Framebuffer(m_renderer->WindowWidth(), m_renderer->WindowHeight());
	bloomBuffer->AddBufferColorAttachment(BUFFER_FLOAT_RGB16); // Albedo color.

	Framebuffer* blurBuffers[2] = { new Framebuffer(m_renderer->WindowWidth(), m_renderer->WindowHeight()), new Framebuffer(m_renderer->WindowWidth(), m_renderer->WindowHeight()) };
	blurBuffers[0]->AddBufferColorAttachment(BUFFER_RGB); // Bright colors.
	blurBuffers[1]->AddBufferColorAttachment(BUFFER_RGB); // Bright colors.

	// Set shaders used for deferred shading pass.
	m_renderer->SetDLightShader(directionalLightSdr);
	m_renderer->SetPLightShader(lightShader);

	float fDeltaTime = 0.0f;

	while(!glfwWindowShouldClose(m_window)) 
	{
		// Time
		auto startTime = std::chrono::high_resolution_clock::now();

		// Quit if escape is pressed.
		if (m_input->GetKey(GLFW_KEY_ESCAPE))
			glfwSetWindowShouldClose(m_window, 1);

		if(m_input->GetKey(GLFW_KEY_UP)) 
		{
			m_v3ParticleTarget += Vector3(0.0f, 0.0f, 10 * -fDeltaTime);
		}
		if (m_input->GetKey(GLFW_KEY_DOWN))
		{
			m_v3ParticleTarget += Vector3(0.0f, 0.0f, 10 * fDeltaTime);
		}
		if (m_input->GetKey(GLFW_KEY_LEFT))
		{
			m_v3ParticleTarget += Vector3(10 * -fDeltaTime, 0.0f, 0.0f);
		}
		if (m_input->GetKey(GLFW_KEY_RIGHT))
		{
			m_v3ParticleTarget += Vector3(10 * fDeltaTime, 0.0f, 0.0f);
		}
		if (m_input->GetKey(GLFW_KEY_PAGE_UP))
		{
			m_v3ParticleTarget += Vector3(0.0f, 10 * fDeltaTime, 0.0f);
		}
		if (m_input->GetKey(GLFW_KEY_PAGE_DOWN))
		{
			m_v3ParticleTarget += Vector3(0.0f, 10 * -fDeltaTime, 0.0f);
		}

		if (m_input->GetKey(GLFW_KEY_1) && m_particleState != EParticleState::PARTICLE_STARS) 
		{
			m_particleState = PARTICLE_STARS;
		}
		else if(m_input->GetKey(GLFW_KEY_2) && m_particleState != EParticleState::PARTICLE_SPHERE_DISSOLVE) 
		{
			m_particleState = PARTICLE_SPHERE_DISSOLVE;
		}
		else if (m_input->GetKey(GLFW_KEY_3) && m_particleState != EParticleState::PARTICLE_BLACK_HOLE)
		{
			m_particleState = PARTICLE_BLACK_HOLE;
		}

		switch (m_particleState)
		{
		case PARTICLE_STARS:
			for (int i = 0; i < STAR_FIELD_COUNT; ++i)
				m_starParticles[i]->Update(fDeltaTime);

			break;
		case PARTICLE_SPHERE_DISSOLVE:

			m_sphereDissolveEffect->Update(fDeltaTime);

			break;

		case PARTICLE_BLACK_HOLE:

			m_blackHoleData.m_target = { m_v3ParticleTarget.x, m_v3ParticleTarget.y, m_v3ParticleTarget.z };
			m_blackHoleEffect->SetGlobalData<GPL_GlobalP3DData>(&m_blackHoleData);

			m_blackHoleEffect->Update(fDeltaTime);

			break;
		}

		GPL_ParticleSystem::Sync();

		// ------------------------------------------------------------------------------------
		// Spear
		modelQuat = glm::tquat<float>();
		modelQuat = glm::rotate(modelQuat, (float)glfwGetTime(), glm::vec3(0.0f, 1.0f, 0.0f));
		modelMatrix = glm::scale(glm::mat4(), glm::vec3(1.0f));
		modelMatrix *= glm::mat4_cast(modelQuat);
		modelMatrix[3] = glm::vec4(2.0f, 0.0f, 0.0f, 1.0f);
		modelMatrix[3][1] = 1.0f;

		// ------------------------------------------------------------------------------------
		// Camera & View matrix

		m_camera.Update(fDeltaTime, m_input, m_window);

		m_renderer->SetViewMatrix(m_camera.GetViewMatrix(), m_camera.GetPosition());

		// ------------------------------------------------------------------------------------

		// Poll events.
		glfwPollEvents();

		// Rendering...

		// Bind G Buffer...

		m_renderer->Start();
		gBuffer->Bind();
		m_renderer->ClearFramebuffer();
		
		// Draw calls here...

		//planeBatch->Add(glm::value_ptr(floorModelMatrix), { 1.0f, 1.0f, 1.0f, 1.0f });

		// Particles...
		particleShader->Use();
		particleShader->SetUniformMat4("model", Matrix4::Identity());

		switch (m_particleState)
		{
		case PARTICLE_STARS:
			for (int i = 0; i < STAR_FIELD_COUNT; ++i)
				m_starParticles[i]->DrawSimple();

			break;
		case PARTICLE_SPHERE_DISSOLVE:

			m_sphereDissolveEffect->DrawSimple();

			break;

		case PARTICLE_BLACK_HOLE:

			m_blackHoleEffect->DrawSimple();

			break;
		}

		Shader::ResetBinding();

		floorMat->DrawStaticMeshes();

		// Flush batches...
		m_renderer->DrawFinal();

		bloomBuffer->Bind();

		// Draw fullscreen quad...
		m_renderer->DrawFSQuad(quadShader, gBuffer->GetTextureArray(), gBuffer->GetAttachmentCount());
		m_renderer->ReportErrors();
		m_renderer->RunDeferredPointLighting(gBuffer->GetTextureArray(), gBuffer->GetAttachmentCount());

		// Bind fullscreen quad for minimal state changes during post effect rendering.
		m_renderer->BindFSQuad();

		// Bind first blur buffer.
		brightColorBuffer->Bind();

		// Render bright color into first hdr
		brightShader->Use();
		m_renderer->BindTextures(bloomBuffer->GetTextureArray(), 1);
		m_renderer->DrawFSQuadNoState();

		Framebuffer* currentBuffer = blurBuffers[1];
		Framebuffer* lastBuffer = blurBuffers[0];

		// Bind first blur buffer.
		lastBuffer->Bind();

		// First horizontal blur pass.
		gaussianHrzntl->Use();
		m_renderer->BindTextures(brightColorBuffer->GetTextureArray(), 1);
		m_renderer->DrawFSQuadNoState();


		// Remaining horizontal passes.
		for(int i = 0; i < BLOOM_PASS_COUNT - 1; ++i)
		{
			currentBuffer->Bind();

			m_renderer->BindTextures(lastBuffer->GetTextureArray(), 1);
			m_renderer->DrawFSQuadNoState();

			Framebuffer* lastBufferCpy = lastBuffer;

			lastBuffer = currentBuffer;
			currentBuffer = lastBufferCpy;
		}

		gaussianVert->Use();

		// Vertical pass.
		for (int i = 0; i < BLOOM_PASS_COUNT; ++i)
		{
			currentBuffer->Bind();

			m_renderer->BindTextures(lastBuffer->GetTextureArray(), 1);
			m_renderer->DrawFSQuadNoState();

			Framebuffer* lastBufferCpy = lastBuffer;

			lastBuffer = currentBuffer;
			currentBuffer = lastBufferCpy;
		}

		// Reset FBO binding.
		m_renderer->ResetFramebufferBinding();

		// Get final textures...
		Texture* m_finalBloomTex[2] = 
		{
			bloomBuffer->GetTextureArray()[0],
			lastBuffer->GetTextureArray()[0]
		};

		// Draw final result...
		bloomHDRShader->Use();
		m_renderer->BindTextures(&m_finalBloomTex[0], 2);
		m_renderer->DrawFSQuadNoState();

		// Unbind fullscreen quad.
		m_renderer->UnbindFSQuad();

		//m_renderer->ReportErrors();
		m_renderer->End();

		// End time...
		auto endTime = std::chrono::high_resolution_clock::now();

		auto timeDuration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();

		fDeltaTime = static_cast<float>(timeDuration) / 1000000.0f;
	}

	// Free rendering related memory.
	delete gBuffer;
	delete brightColorBuffer;

	delete bloomBuffer;

	delete blurBuffers[1];
	delete blurBuffers[0];

	delete quadShader;
	delete brightShader;
	delete bloomHDRShader;
	delete particleShader;

	delete gaussianVert;
	delete gaussianHrzntl;

	delete directionalLightSdr;
	delete lightShader;
	delete planeShader;

	delete floorMat;

	delete planeMesh;
	delete sphereMesh;

	// Free particle systems and behaviours.
	delete m_blackHoleEffect;

	delete m_sphereDissolveEffect;

	for (int i = 0; i < STAR_FIELD_COUNT; ++i)
		delete m_starParticles[i];

	delete m_blackHoleBehaviour;
	delete m_sphereDissolveBehaviour;
	delete m_starBehaviour;
}

void Application::ErrorCallBack(int error, const char* desc)
{
	std::cout << "GLFW Error: " << desc << "\n";
}

void Application::KeyCallBack(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	m_input->GetCurrentState()[key] = action;
}

void Application::MouseButtonCallBack(GLFWwindow* window, int button, int action, int mods)
{
	m_input->GetCurrentMouseState()->m_buttons[button] = action - 1;
}

void Application::CursorPosCallBack(GLFWwindow* window, double dXPos, double dYPos)
{
	MouseState* currentState = m_input->GetCurrentMouseState();

	currentState->m_fMouseAxes[0] = dXPos;
	currentState->m_fMouseAxes[1] = dYPos;
}

void Application::MouseScrollCallBack(GLFWwindow* window, double dXOffset, double dYOffset)
{
	MouseState* currentState = Input::GetInstance()->GetCurrentMouseState();

	currentState->m_fMouseAxes[2] = dXOffset;
	currentState->m_fMouseAxes[3] = dYOffset;
}
