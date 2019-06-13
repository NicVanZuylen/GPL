#version 440 core

layout (location = 0) in vec4 vertPos;
layout (location = 1) in vec4 normal;
layout (location = 2) in vec2 texCoords;
layout (location = 3) in vec4 color;
layout (location = 4) in mat4 model;

layout (std140) uniform GlobalMatrices
{
    mat4 view;
    mat4 projection;
};

out vec4 modelColor;
out vec4 modelNormal;
out vec4 fragPos;
out vec2 modelTexCoords;

void main() 
{
    modelColor = color;
	modelTexCoords = vertPos.xy;
	modelNormal = model * normal;
	
	fragPos = model * vertPos;

    gl_Position = projection * view * model * vertPos;
	
	//gl_Position = model * view * projection * vertPos;
	
	//gl_Position = vertPos;
}