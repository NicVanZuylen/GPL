#version 440 core

layout (location = 0) in vec4 vertPos;
layout (location = 1) in vec4 normal;
layout (location = 2) in vec4 tangent;
layout (location = 3) in vec2 texCoords;
layout (location = 4) in vec3 position;
layout (location = 5) in float scale;
layout (location = 6) in vec3 color;

uniform mat4 model;

layout (std140) uniform GlobalMatrices
{
    mat4 view;
    mat4 projection;
};

out vec3 modelColor;
out float modelScale;
out vec4 fragPos;

void main() 
{
    // Pass to next stage...
	modelColor = color;
	
	modelScale = scale;
	
    // Model matrix...
	mat4 modelCpy = model;

	modelCpy[0] *= scale;
	modelCpy[1] *= scale;
	modelCpy[2] *= scale;

	modelCpy[3][0] = position.x;
	modelCpy[3][1] = position.y;
	modelCpy[3][2] = position.z;

	fragPos = modelCpy * vertPos; // Get worldspace fragment position.
	
    gl_Position = projection * view * modelCpy * vertPos;
}