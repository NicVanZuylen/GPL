#version 440 core

in vec4 lightColorRadius;
in vec3 lightPos;
in vec2 fragPos;
out vec4 fragColor;

uniform sampler2D textureMaps[16];

// Specular strength
uniform float specularStrength;

layout (std140) uniform GlobalMatrices
{
    mat4 view;
    mat4 projection;
    vec3 viewPos;
};

#define CONSTANT 1.0f
#define LINEAR_CONSTANT 0.7f * 2.0f / 0.5f
#define QUADRATIC_CONSTANT 1.8f * 2.0f / 0.5f

#define TEMP_SPEC_STRENGTH 0.5f

void main() 
{
    float lightRadius = lightColorRadius.w;
	vec4 lightColor = vec4(lightColorRadius.rgb, 1.0f);
	
	ivec2 frameDimensions = textureSize(textureMaps[0], 0);
	vec2 fragTexCoord = gl_FragCoord.xy;
	fragTexCoord.x /= float(frameDimensions.x);
	fragTexCoord.y /= float(frameDimensions.y);
	
	// Sample texels...
    vec4 diffuse = texture(textureMaps[0], fragTexCoord);
	vec4 position = texture(textureMaps[1], fragTexCoord);
	vec4 normal = texture(textureMaps[2], fragTexCoord);
	vec4 specularTexel = texture(textureMaps[3], fragTexCoord);
	
	// Distance between the light and fragment position.
	vec3 posDiff = lightPos - position.xyz;
	vec3 lightDir = normalize(posDiff);
	float dist = length(posDiff);

	// View direction and reflection vector.
	vec3 viewDir = normalize(viewPos - position.xyz);
    vec3 reflectionDir = reflect(-lightDir, normal.xyz);

	// Attenuation function...
	float attenuation = clamp(1.0f - dist / lightRadius, 0.0f, 1.0f);
	attenuation *= attenuation;

    // Ambient component
	vec3 lighting = diffuse.rgb * 0.1f; // 0.1 will be replaced with ambient color.

	// Dot product between surface normal and the direction towards the light.
	float normalDot = max(dot(normal.xyz, lightDir), 0.0f);

	// Calculate specular.
	float spec = pow(max(dot(viewDir, reflectionDir), 0.0f), specularTexel.a);
    vec3 finalSpec = spec * TEMP_SPEC_STRENGTH * specularTexel.rgb;

    // Add to light level with attenuation, diffuse and light color.
	lighting += normalDot * attenuation * diffuse.rgb * lightColor.rgb;
	lighting += finalSpec; // Add specular component.

	// Output.
	fragColor = vec4(lighting, 1.0f);
}