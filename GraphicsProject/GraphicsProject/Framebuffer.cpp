#include "FrameBuffer.h"
#include "glad\glad.h"
#include "Renderer.h"
#include "Texture.h"
#include <iostream>

Framebuffer::Framebuffer(int nWidth, int nHeight) 
{
	m_attachmentTextures = new Texture*[MAX_COLOR_ATTACHMENT_COUNT];
	m_depthTexture = nullptr;
	m_nAttachmentCount = 0;
	m_glBufferHandle = 0;

	m_nWidth = nWidth;
	m_nHeight = nHeight;

	// Create framebuffer.
	glGenFramebuffers(1, &m_glBufferHandle);
	glBindFramebuffer(GL_FRAMEBUFFER, m_glBufferHandle);

	// Blend function...
	glBlendFunci(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_FRAMEBUFFER);

	// Unbind buffers.
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glBindTexture(GL_TEXTURE_2D, 0);
}

Framebuffer::~Framebuffer() 
{
	// Delete texture objects.
	for (unsigned int i = 0; i < m_nAttachmentCount; ++i)
		delete m_attachmentTextures[i];

	delete[] m_attachmentTextures;

	if (m_depthTexture)
		delete m_depthTexture;

	// Delete all textures and the framebuffer.
	glDeleteFramebuffers(1, &m_glBufferHandle);
	glDeleteTextures(m_nAttachmentCount, m_glTextureHandles);
}

void Framebuffer::Bind() 
{
	glBindFramebuffer(GL_FRAMEBUFFER, m_glBufferHandle);
}

void Framebuffer::AddBufferColorAttachment(EAttachmentFormat eFormat)
{
	if (m_nAttachmentCount >= MAX_COLOR_ATTACHMENT_COUNT)
		return;

	glBindFramebuffer(GL_FRAMEBUFFER, m_glBufferHandle);

	// Get unused handle.
	unsigned int& nHandle = m_glTextureHandles[m_nAttachmentCount];

	// Generate texture buffer.
	glGenTextures(1, &nHandle);
	glBindTexture(GL_TEXTURE_2D, nHandle);

	// Allocate texture buffer memory.
	switch (eFormat) 
	{
	case BUFFER_RGB:
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, m_nWidth, m_nHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);
		break;

	case BUFFER_RGBA:
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, m_nWidth, m_nHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
		break;

	case BUFFER_FLOAT_RGB16:
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, m_nWidth, m_nHeight, 0, GL_RGB, GL_FLOAT, 0);
		break;

	case BUFFER_FLOAT_RGB32:
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, m_nWidth, m_nHeight, 0, GL_RGB, GL_FLOAT, 0);
		break;

	case BUFFER_FLOAT_RGBA16:
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, m_nWidth, m_nHeight, 0, GL_RGBA, GL_FLOAT, 0);
		break;

	case BUFFER_FLOAT_RGBA32:
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_nWidth, m_nHeight, 0, GL_RGBA, GL_FLOAT, 0);
		break;
	}

	// Disable texture wrapping for framebuffers.
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	// Texture filtering.
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); // Nearest neighbour on shrink.
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); // Nearest neighbour on expand.

	// Remember attachment format.
	m_eAttachmentFormats[m_nAttachmentCount] = eFormat;

	// Create texture object for external use.
	m_attachmentTextures[m_nAttachmentCount] = new Texture(nHandle, m_nWidth, m_nHeight);

	// Attach texture to framebuffer.
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + m_nAttachmentCount, GL_TEXTURE_2D, nHandle, 0);

	// Set draw buffers.
	unsigned int drawBuffers[MAX_COLOR_ATTACHMENT_COUNT] = 
	{
		GL_COLOR_ATTACHMENT0,
		GL_COLOR_ATTACHMENT1,
		GL_COLOR_ATTACHMENT2,
		GL_COLOR_ATTACHMENT3,
		GL_COLOR_ATTACHMENT4,
		GL_COLOR_ATTACHMENT5,
		GL_COLOR_ATTACHMENT6,
		GL_COLOR_ATTACHMENT7
	};

	glDrawBuffers(m_nAttachmentCount + 1, drawBuffers);

	unsigned int bufferStatus = glCheckFramebufferStatus(GL_FRAMEBUFFER);

	// Unbind framebuffer and frame texture.
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glBindTexture(GL_TEXTURE_2D, 0);

	// Increment attachment count.
	++m_nAttachmentCount;
}

void Framebuffer::AddDepthAttachment() 
{
	glBindFramebuffer(GL_FRAMEBUFFER, m_glBufferHandle);

	// Generate texture buffer.
	glGenTextures(1, &m_glDepthTextureHandle);
	glBindTexture(GL_TEXTURE_2D, m_glDepthTextureHandle);

	// Fill depth buffer.
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, m_nWidth, m_nHeight, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, 0);

	// Texture filtering.
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); // Nearest neighbour on shrink.
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); // Nearest neighbour on expand.

	// Create texture object for external use.
	if (m_depthTexture)
		delete m_depthTexture;

	m_depthTexture = new Texture(m_glDepthTextureHandle, m_nWidth, m_nHeight);

	// Attach texture to framebuffer.
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, m_glDepthTextureHandle, 0);

	// Unbind framebuffer and frame texture.
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glBindTexture(GL_TEXTURE_2D, 0);
}

void Framebuffer::ResizeBuffers(int nWidth, int nHeight) 
{
	m_nWidth = nWidth;
	m_nHeight = nHeight;

	// Resize color attachments...
	for(unsigned int i = 0; i < m_nAttachmentCount; ++i) 
	{
		glBindTexture(GL_TEXTURE_2D, m_glTextureHandles[i]);

		// Resize texture buffer.
		switch (m_eAttachmentFormats[i])
		{
		case BUFFER_RGB:
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, m_nWidth, m_nHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);
			break;

		case BUFFER_RGBA:
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, m_nWidth, m_nHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
			break;

		case BUFFER_FLOAT_RGB16:
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, m_nWidth, m_nHeight, 0, GL_RGB, GL_FLOAT, 0);
			break;

		case BUFFER_FLOAT_RGB32:
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, m_nWidth, m_nHeight, 0, GL_RGB, GL_FLOAT, 0);
			break;

		case BUFFER_FLOAT_RGBA16:
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, m_nWidth, m_nHeight, 0, GL_RGBA, GL_FLOAT, 0);
			break;

		case BUFFER_FLOAT_RGBA32:
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_nWidth, m_nHeight, 0, GL_RGBA, GL_FLOAT, 0);
			break;
		}
	}
}

unsigned int Framebuffer::GetFrameBufferHandle()
{
	return m_glBufferHandle;
}

unsigned int Framebuffer::GetTextureHandle(int nAttachmentIndex) 
{
	return m_glTextureHandles[nAttachmentIndex % MAX_COLOR_ATTACHMENT_COUNT];
}

Texture** Framebuffer::GetTextureArray() 
{
	return m_attachmentTextures;
}

Texture* Framebuffer::GetDepthTexture() 
{
	return m_depthTexture;
}

unsigned int Framebuffer::GetAttachmentCount() 
{
	return m_nAttachmentCount;
}