#version 440 core

layout (location = 0) out vec4 fragOut; // Frag out! take cover!

in vec2 texCoords;

uniform sampler2D textureMaps[16];

void main() 
{
    vec4 fragColor = texture(textureMaps[0], texCoords) + (texture(textureMaps[5], texCoords) * 20.0f);

    fragOut = vec4(fragColor.rgb * 0.1f, 1.0f);

	gl_FragDepth = 1.0f;
}