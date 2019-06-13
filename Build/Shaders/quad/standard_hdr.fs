#version 440 core

layout (location = 0) out vec4 fragOut; // Frag out! take cover!

in vec2 texCoords;

uniform sampler2D textureMaps[16];

uniform float exposure;

void main() 
{
    vec3 fragColor = texture(textureMaps[0], texCoords).rgb;

    const float gamma = 1.1f;

    // Exposure tone mapping
    //vec3 mappedColor = vec3(1.0f) - exp(-fragColor * exposure);
    vec3 mappedColor = fragColor;

    // Gamma correction
    mappedColor = pow(mappedColor, vec3(1.0f / gamma));

    // Output
    fragOut = vec4(mappedColor, 1.0f);
}