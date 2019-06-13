#version 440 core

layout (location = 0) out vec4 fragOut; // Frag out! take cover!

in vec2 texCoords;

uniform sampler2D textureMaps[16];

void main() 
{
    vec4 fragColor = texture(textureMaps[0], texCoords);
    vec4 bloomBlur = texture(textureMaps[1], texCoords);
	
	fragOut = vec4(fragColor.rgb + bloomBlur.rgb, 1.0f);
}