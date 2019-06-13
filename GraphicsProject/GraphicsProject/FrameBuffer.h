#pragma once

#define MAX_COLOR_ATTACHMENT_COUNT 8

class Texture;

enum EAttachmentFormat 
{
	BUFFER_RGB = 0x1907,
	BUFFER_RGBA = 0x1908,
	BUFFER_FLOAT_RGBA16 = 0x881A,
	BUFFER_FLOAT_RGBA32 = 0x8814,
	BUFFER_FLOAT_RGB16 = 0x881B,
	BUFFER_FLOAT_RGB32 = 0x8815
};

class Framebuffer 
{
public:

	Framebuffer(int nWidth, int nHeight);

	~Framebuffer();

	/*
	Description: Bind this framebuffer.
	*/
	void Bind();

	/*
	Description: Add an attachment buffer to this framebuffer such as a diffuse or normal buffer.
	Param:
	    EAttachmentFormat eFormat: The format of attachment.
	*/
	void AddBufferColorAttachment(EAttachmentFormat eFormat);

	/*
	Description: Add a depth attachment to this framebuffer to allow for depth testing.
	*/
	void AddDepthAttachment();

	/*
	Description: Change the width and height of the framebuffer.
	*/
	void ResizeBuffers(int nWidth, int nHeight);

	/*
	Description: Get the the OpenGL handle for this framebuffer.
	*/
	unsigned int GetFrameBufferHandle();

	/*
	Description: Get the OpenGL handle for this framebuffer's texture.
	Param:
	    int nAttachmentIndex: The index of the color attachment texture of this framebuffer.
	*/
	unsigned int GetTextureHandle(int nAttachmentIndex);

	/*
	Description: Get the array of color attachement textures present within this framebuffer.
	Return Type: Texture**
	*/
	Texture** GetTextureArray();

	/*
	Description: Get the depth buffer texture of this framebuffer.
	Return Type: Texture*
	*/
	Texture* GetDepthTexture();

	/*
	Description: Get the amount of attachments present within this framebuffer.
	Return Type: unsigned int
	*/
	unsigned int GetAttachmentCount();

private:

	// Texture handles.
	unsigned int m_glTextureHandles[MAX_COLOR_ATTACHMENT_COUNT];
	unsigned int m_glDepthTextureHandle;
	EAttachmentFormat m_eAttachmentFormats[MAX_COLOR_ATTACHMENT_COUNT];
	// Texture objects.
	Texture** m_attachmentTextures;
	Texture* m_depthTexture;

	unsigned int m_nAttachmentCount;
	unsigned int m_glBufferHandle;

	int m_nWidth;
	int m_nHeight;
};
