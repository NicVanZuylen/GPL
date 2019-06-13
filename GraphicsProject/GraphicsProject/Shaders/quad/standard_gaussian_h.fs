#version 440 core

layout (location = 0) out vec3 fragOut; // Frag out! take cover!

in vec2 texCoords;

uniform sampler2D textureMaps[16];

uniform float power;
uniform float weights[5] = float[] (0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

void main() 
{
    vec3 fragColor = texture(textureMaps[0], texCoords).rgb * weights[0] * power;
	vec2 texelSize = vec2(1.0f) / textureSize(textureMaps[0], 0);
	texelSize.y = 0.0f;
	
	vec2 finalOffset = texelSize;
	
	// Blur sampling...
	fragColor += texture(textureMaps[0], texCoords + finalOffset).rgb * weights[0];
	fragColor += texture(textureMaps[0], texCoords - finalOffset).rgb * weights[0];
	
	finalOffset = texelSize * 2;
	
	fragColor += texture(textureMaps[0], texCoords + finalOffset).rgb * weights[1];
	fragColor += texture(textureMaps[0], texCoords - finalOffset).rgb * weights[1];
	
	finalOffset = texelSize * 3;
	
	fragColor += texture(textureMaps[0], texCoords + finalOffset).rgb * weights[2];
	fragColor += texture(textureMaps[0], texCoords - finalOffset).rgb * weights[2];
	
	finalOffset = texelSize * 4;
	
	fragColor += texture(textureMaps[0], texCoords + finalOffset).rgb * weights[3];
	fragColor += texture(textureMaps[0], texCoords - finalOffset).rgb * weights[3];
	
	finalOffset = texelSize * 5;
	
	fragColor += texture(textureMaps[0], texCoords + finalOffset).rgb * weights[4];
	fragColor += texture(textureMaps[0], texCoords - finalOffset).rgb * weights[4];
	
	fragOut = fragColor;
}