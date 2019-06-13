#version 330 core

in vec4 spriteColor;
in vec4 fragPos;
in vec2 texCoords;
in float textureLocation;

#define TEXTURE_BATCH_SIZE 16

uniform sampler2D textureBatch[TEXTURE_BATCH_SIZE];

layout (location = 0) out vec4 fragDiffuseOut;
layout (location = 1) out vec4 fragPosOut;

void main()
{
    vec4 diffuseTexel = texture(textureBatch[int(textureLocation)], texCoords) * spriteColor;


	//gl_FragDepth = gl_FragDepth + 1.0f;

	fragDiffuseOut = diffuseTexel;
	fragPosOut = fragPos;
	fragPosOut.a = 1.0f;
}