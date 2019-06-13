#version 330 core

layout (location = 0) out vec4 diffuse;

in vec2 texCoords;

uniform sampler2D textureBatch[16];

void main() 
{
    diffuse = vec4(1.0f - texture(textureBatch[0], texCoords).xyz, 1.0f);
}