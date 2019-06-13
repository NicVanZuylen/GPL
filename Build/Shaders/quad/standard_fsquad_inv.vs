#version 330 core

layout (location = 0) in vec2 vertex;
layout (location = 1) in vec2 texCoord;

out vec2 texCoords;

void main() 
{
    texCoords = texCoord;

	gl_Position = vec4(vertex, 0.0f, 1.0f);
}