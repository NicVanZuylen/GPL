#version 440 core

in vec4 modelColor;
in vec4 fragPos;
in vec2 modelTexCoords;
in vec3 modelNormal;
in float shininess;

uniform sampler2D textureMaps[16];

layout (location = 0) out vec4 fragDiffuseOut;
layout (location = 1) out vec4 fragPositionOut;
layout (location = 2) out vec4 fragNormalOut;
layout (location = 3) out vec4 fragSpecularOut;
layout (location = 4) out vec3 fragRoughnessOut;
layout (location = 5) out vec3 fragEmissionOut;

void main() 
{
    fragDiffuseOut = vec4(1.0f);
	
	// Output position.
    fragPositionOut = vec4(fragPos.xyz, 1.0f); // Position
	
    vec3 finalNormal = modelNormal;
	
	// Output normal
    fragNormalOut = vec4(finalNormal, 1.0f);

	// Output specular.
	fragSpecularOut = vec4(1.0f);
	
	// Output roughness
	fragRoughnessOut = vec3(0.3f, 0.0f, 0.5f);
	
	// Output emission
	fragEmissionOut = vec3(0.0f);
}