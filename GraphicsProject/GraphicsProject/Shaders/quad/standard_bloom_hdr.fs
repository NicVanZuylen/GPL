#version 440 core

layout (location = 0) out vec3 fragOut;

in vec2 texCoords;

uniform sampler2D textureMaps[16];

uniform float exposure;

void main() 
{
    vec3 fragColor = texture(textureMaps[0], texCoords).rgb;
	vec3 fragBloomBlur = texture(textureMaps[1], texCoords).rgb;
    
	// Exposure tone mapping
	vec3 mappedColor = vec3(1.0f) - exp(-(fragColor + fragBloomBlur) * exposure);
	
	// Gamma correction
    mappedColor = pow(mappedColor, vec3(1.0f / 1.2f));
	
	fragOut = mappedColor;
}