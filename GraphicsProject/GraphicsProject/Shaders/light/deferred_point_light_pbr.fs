#version 440 core

in vec4 lightColorRadius;
in vec3 lightPos;
in vec2 fragPos;

out vec4 fragColor;

uniform sampler2D textureMaps[16];

layout (std140) uniform GlobalMatrices
{
    mat4 view;
    mat4 projection;
    vec3 viewPos;
};

#define CONSTANT 1.0f
#define LINEAR_CONSTANT 0.7f * 2.0f / 0.5f
#define QUADRATIC_CONSTANT 1.8f * 2.0f / 0.5f

//#define TEMP_SPEC_STRENGTH 0.5f
//#define TEMP_REFLECTION_COEFICCIENT 0.5f
//#define TEMP_ROUGHNESS 0.3f

#define PI 3.1415926535897932384626433832f
#define E  2.71828182845904523536028747135f

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
	
    normal.xyz = normalize(normal.xyz);

	vec4 miscData = texture(textureMaps[4], fragTexCoord);
	
	float roughness = miscData.r;
	float specularStrength = miscData.g;
	float reflectionCoefficient = miscData.b;
	
	// -----------------------------------------------------------------------------
	// Vectors
	// Distance between the light and fragment position.
	vec3 posDiff = lightPos - position.xyz;
	vec3 lightDir = normalize(posDiff);
	float dist = length(posDiff);

	// View direction and reflection vector.
	//vec3 viewDir = normalize(-position.xyz);
	vec3 viewDir = normalize(viewPos - position.xyz);
    vec3 reflectionDir = reflect(-lightDir, normal.xyz);

	// -----------------------------------------------------------------------------
	// Attenuation
	
	// Attenuation function...
	float attenuation = clamp(1.0f - dist / lightRadius, 0.0f, 1.0f);
	attenuation *= attenuation;

	// -----------------------------------------------------------------------------
	// Physically Based Lighting
	
	// ---------------- Oren Nayar Diffuse -----------------
	float roughnessSqr = roughness * roughness;
	
	float A = 1.0f - 0.5f * (roughnessSqr / (roughnessSqr + 0.33f));
	float B = 0.45f * (roughnessSqr / (roughnessSqr + 0.09f)); 
	
	float normalDotLight = max(dot(normal.xyz, lightDir), 0.0f);
	float normalDotView = max(dot(normal.xyz, viewDir), 0.0f);
	
	vec3 lightProj = normalize((lightDir - normal.xyz) * normalDotLight);
    vec3 viewProj = normalize((viewDir - normal.xyz) * normalDotView); 
	
	float projDot = max(0.0f, dot(lightProj, viewProj));
	
	float alpha = sin(max(acos(normalDotView), acos(normalDotLight)));
	float beta = tan(min(acos(normalDotView), acos(normalDotLight)));
	
	float orenNayar = normalDotLight * (A + (B * projDot * alpha * beta));
	
	// ---------------- Cook Torrence Specular -----------------
	
	vec3 halfVec = normalize(lightDir + viewDir);
	
	float reflectNormalDotView = max(dot(reflectionDir, viewDir), 0.0f);
	
	float normalDotHalf = max(dot(normal.xyz, halfVec), 0.0f);
	float normalDotHalfSqr = normalDotHalf * normalDotHalf;
	
	float exponent = -(1 - normalDotHalfSqr) / (normalDotHalfSqr * roughnessSqr);
	float D = exp(exponent) / (roughnessSqr * (normalDotHalfSqr * normalDotHalfSqr));
	
	float F = reflectionCoefficient + (1.0f - reflectionCoefficient) * pow(1.0f - reflectNormalDotView, 5);
	
	float frac = 2.0f * normalDotHalf / dot(viewDir, halfVec);
	float G = min(1, min(frac * reflectNormalDotView, frac * normalDotLight));
	
	float cookTorrence = max((D * F * G) / (reflectNormalDotView * PI), 0.0f);
	
	// -----------------------------------------------------------------------------
	// Lighting
	
    // Ambient component
	vec3 lighting = diffuse.rgb * 0.1f; // 0.1 will be replaced with ambient color.

	// Calculate specular.
	float spec = pow(max(dot(viewDir, reflectionDir), 0.0f), specularTexel.a);
    vec3 finalSpec = cookTorrence * specularStrength * specularTexel.rgb;

    // Add to light level with attenuation, diffuse and light color.
	lighting += orenNayar * diffuse.rgb * lightColor.rgb;
	lighting += finalSpec; // Add specular component.

	// Output.
	fragColor = vec4(lighting * attenuation, 1.0f);
	
	// -----------------------------------------------------------------------------
}