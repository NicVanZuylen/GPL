#version 440 core

layout (location = 0) in vec4 vertPos;
layout (location = 1) in vec4 normal;
layout (location = 2) in vec4 tangent;
layout (location = 3) in vec2 texCoords;
layout (location = 4) in vec4 color; // Radius in w.
layout (location = 5) in mat4 model;

layout (std140) uniform GlobalMatrices
{
    mat4 view;
    mat4 projection;
    vec3 viewPos;
};

out vec4 lightColorRadius;
out vec3 lightPos;
out vec2 fragPos;

void main() 
{
    lightColorRadius = color;
	
	lightPos = vec3(model[3][0], model[3][1], model[3][2]);
	
	vec4 outPos = projection * view * model * vertPos;
	
	fragPos = outPos.xy;
	
	gl_Position = outPos;
}