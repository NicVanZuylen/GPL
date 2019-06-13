#pragma once

class Texture 
{
public:

	Texture(const char* szFilePath);

	Texture(unsigned int glTextureHandle, int nWidth, int nHeight);

	~Texture();

	/*
	Description: Bind this texture to a GPU texture unit.
	*/
	void Bind();

	/*
	Description: Get the OpenGL handle for this texture.
	Return Type: unsigned int
	*/
	unsigned int GetHandle();

	/*
	Description: Get the width in pixels of the texture.
	Return Type: int
	*/
	int GetWidth();

	/*
	Description: Get the height in pixels of the texture.
	Return Type: int
	*/
	int GetHeight();

protected:

	unsigned char* m_data;
	unsigned int m_glHandle;
	int m_nWidth;
	int m_nHeight;
	int m_nChannels;
	bool m_bOwnsTexture;
};