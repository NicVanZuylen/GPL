#version 440 core

layout (location = 0) in vec4 vertPos;
layout (location = 1) in vec4 normal;
layout (location = 2) in vec4 tangent;
layout (location = 3) in vec2 texCoords;
layout (location = 4) in vec4 color;
layout (location = 5) in mat4 model;
layout (location = 9) in mat3 normalMat;

layout (std140) uniform GlobalMatrices
{
    mat4 view;
    mat4 projection;
};

uniform float specularShininess;

out vec4 modelColor;
out vec4 fragPos;
out vec3 modelNormal;
out vec2 modelTexCoords;
out float shininess;

void main() 
{
    // Pass to next stage...
    modelColor = color;
	modelNormal = normal.xyz;
	modelTexCoords = texCoords * 2;
	shininess = specularShininess;
	
	fragPos = model * vertPos; // Get worldspace fragment position.

    gl_Position = projection * view * model * vertPos;
}