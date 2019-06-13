#version 440 core

layout (location = 0) out vec3 fragBrightOut;

in vec2 texCoords;

uniform sampler2D textureMaps[16];

void main() 
{
    vec4 fragColor = texture(textureMaps[0], texCoords);
    
	// Brightness testing...
	float brightness = dot(fragColor.rgb, vec3(0.2126f, 0.7152f, 0.0722f));
	
	fragBrightOut = fragColor.rgb * int(brightness > 1.0f);
}