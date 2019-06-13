#version 440 core

layout (location = 0) in vec4 vertPos;
layout (location = 1) in vec4 normal;
layout (location = 2) in vec4 tangent;
layout (location = 3) in vec2 texCoords;

layout (location = 4) in mat4 model;
layout (location = 5) in vec3 color;
layout (location = 6) in float radius;

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
	lightColorRadius = vec4(color, radius);

	lightPos = vec3(model[3][0], model[3][1], model[3][2]);
	
	vec4 outPos = projection * view * model * vertPos;
	
	fragPos = outPos.xy;
	
	gl_Position = outPos;
}