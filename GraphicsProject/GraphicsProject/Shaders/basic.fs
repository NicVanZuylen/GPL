#version 440 core

in vec4 modelColor;
in vec4 modelTexCoords;

uniform sampler2D textureMaps[16];

out vec4 fragColor;

void main() 
{
    fragColor = texture(textureMaps[0], modelTexCoords.xy);
}