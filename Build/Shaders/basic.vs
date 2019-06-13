#version 440 core

layout (location = 0) in vec4 vertPos;
layout (location = 1) in vec4 texCoords;
layout (location = 2) in vec4 color;
layout (location = 3) in mat4 model;

layout (std140) uniform GlobalMatrices
{
    mat4 view;
    mat4 projection;
};

out vec4 modelColor;
out vec4 modelTexCoords;

void main() 
{
    modelColor = color;
	modelTexCoords = texCoords;

    gl_Position = projection * view * model * vertPos;
	//gl_Position = vertPos;
}