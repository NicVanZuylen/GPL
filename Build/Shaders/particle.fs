#version 440 core

in vec3 modelColor;
in float modelScale;
in vec4 fragPos;

layout (location = 0) out vec4 fragDiffuseOut;

void main() 
{
    if(modelScale == 0.0f)
	    discard;

	// Output emission
	fragDiffuseOut = vec4(modelColor, 1.0f);
}