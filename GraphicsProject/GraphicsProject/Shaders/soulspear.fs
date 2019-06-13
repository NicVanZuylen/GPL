#version 440 core

in mat3 tbnMat;
in vec4 modelColor;
in vec4 fragPos;
in vec2 modelTexCoords;
in float shininess;

uniform sampler2D textureMaps[16];

layout (location = 0) out vec4 fragDiffuseOut;
layout (location = 1) out vec4 fragPositionOut;
layout (location = 2) out vec4 fragNormalOut;
layout (location = 3) out vec4 fragSpecularOut;
layout (location = 4) out vec3 fragRoughnessOut;

void main() 
{
    fragDiffuseOut = texture(textureMaps[0], modelTexCoords); // Diffuse
	
	// Output position.
    fragPositionOut = vec4(fragPos.xyz, 1.0f); // Position
	
	// Read normal map and translate normal vector from tangent space to worldspace using the TBN matrix.
	vec3 finalNormal = tbnMat * normalize(texture(textureMaps[1], modelTexCoords).xyz * 2.0f - 1.0f); // Normal

	// Output normal.
	fragNormalOut = vec4(finalNormal, 1.0f);
	
	// Output specular.
	fragSpecularOut = vec4(texture(textureMaps[2], modelTexCoords).rgb, shininess);
	
	// Output roughness
	fragRoughnessOut = vec3(0.2f, 0.6f, 1.0f);
}