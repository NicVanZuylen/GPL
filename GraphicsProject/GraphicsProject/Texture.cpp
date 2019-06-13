#include "Texture.h"
#include "GLAD/glad.h"
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

Texture::Texture(const char* szFileName) 
{
	m_data = nullptr;

	m_nWidth = 0;
	m_nHeight = 0;
	m_nChannels = 0;
	m_glHandle = 0;
	m_bOwnsTexture = true;

	if (!szFileName)
		return;

	// Load image...
	m_data = stbi_load(szFileName, &m_nWidth, &m_nHeight, &m_nChannels, STBI_rgb_alpha);

	if(m_data) 
	{
		// Create and bind texture buffer.s
		glGenTextures(1, &m_glHandle);
		glBindTexture(GL_TEXTURE_2D, m_glHandle);

		// Send data to buffer.
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, m_nWidth, m_nHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, m_data);

		// Generate mipmap
		glGenerateMipmap(GL_TEXTURE_2D);

		// Specify texture parameters...

		// Texture wrapping
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

		// Texture filtering
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); // Nearest neighbour on shrink.
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); // Nearest neighbour on expand.

		// Unbind texture when finished.
		glBindTexture(GL_TEXTURE_2D, 0);

		std::cout << "Successfully loaded image: " << szFileName << std::endl;
	}
	else
	{
		std::cout << "Failed to load image: " << szFileName << std::endl;
	}
}

Texture::Texture(unsigned int glTextureHandle, int nWidth, int nHeight)
{
	m_bOwnsTexture = false;

	m_glHandle = glTextureHandle;
	m_data = nullptr;
	m_nChannels = 0;

	m_nWidth = nWidth;
	m_nHeight = nHeight;
}

Texture::~Texture() 
{
	if (m_data && m_bOwnsTexture)
		stbi_image_free(m_data);

	if (m_glHandle > 0 && m_bOwnsTexture)
	{
		glDeleteTextures(1, &m_glHandle);
		m_glHandle = 0;
	}
}

void Texture::Bind() 
{
	glBindTexture(GL_TEXTURE_2D, m_glHandle);
}

unsigned int Texture::GetHandle() 
{
	return m_glHandle;
}

int Texture::GetWidth() 
{
	return m_nWidth;
}

int Texture::GetHeight() 
{
	return m_nHeight;
}