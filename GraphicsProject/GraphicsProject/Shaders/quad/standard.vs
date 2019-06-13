#version 330 core

layout (location = 0) in vec2 vertex;
layout (location = 1) in vec2 texCoord;
layout (location = 2) in vec4 color;
layout (location = 3) in vec4 uvRect;
layout (location = 4) in mat4 model;
		
out vec4 spriteColor;
out vec4 fragPos;
out vec2 texCoords;
out float textureLocation;

layout (std140) uniform GlobalMatrices
{
    mat4 projection;
};

void main()
{
    spriteColor = color;

    // Apply UV rectangle transformations to the default texture coordinate.
    texCoords = (texCoord * uvRect.zw) + uvRect.xy;

    // Texture location...
	textureLocation = model[3][2];

	// Vertex position:

	// Worldspace position of fragment
	fragPos = model * vec4(vertex, 0.0f, 1.0f);

	// Screen space normalized coordinates of the fragment.
	gl_Position = projection * fragPos;
}