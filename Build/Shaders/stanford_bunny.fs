#version 440 core

in vec4 modelColor;
in vec4 modelNormal;
in vec4 fragPos;
in vec2 modelTexCoords;

uniform sampler2D textureMaps[16];

layout (location = 0) out vec4 fragDiffuseOut;
layout (location = 1) out vec4 fragPositionOut;
layout (location = 2) out vec4 fragNormalOut;

void main() 
{
    //fragDiffuseOut = texture(textureMaps[0], modelTexCoords);
	fragDiffuseOut = vec4(1.0f);
    fragPositionOut = vec4(fragPos.xyz, 1.0f);
	fragNormalOut = modelNormal;
	//fragColor = vec4(1.0f);
}