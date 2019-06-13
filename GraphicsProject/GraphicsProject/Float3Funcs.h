#ifndef FLOAT3_FUNCS
#define FLOAT3_FUNCS

#include <vector_types.h>

__device__ __forceinline__ float3 Add(const float3& a, const float3& b) 
{
	return { a.x + b.x, a.y + b.y, a.z + b.z };
}

__device__ __forceinline__ float3 Sub(const float3& a, const float3& b) 
{
	return { a.x - b.x, a.y - b.y, a.z - b.z };
}

__device__ __forceinline__ float3 Mul(const float3& a, const float& b) 
{
	return { a.x * b, a.y * b, a.z * b };
}

__device__ __forceinline__ float3 Div(const float3& a, const float& b)
{
	return { a.x / b, a.y / b, a.z / b };
}

__device__ __forceinline__ float3 Mul3(const float3& a, const float3& b)
{
	return { a.x * b.x, a.y * b.y, a.z * b.z };
}

__device__ __forceinline__ float3 Div3(const float3& a, const float3& b)
{
	return { a.x / b.x, a.y / b.y, a.z / b.z };
}

__device__ __forceinline__ float Magnitude(const float3& f) 
{
	return sqrtf((f.x * f.x) + (f.y * f.y) + (f.z * f.z));
}

__device__ __forceinline__ float SqrMagnitude(const float3& f)
{
	return (f.x * f.x) + (f.y * f.y) + (f.z * f.z);
}

__device__ __forceinline__ float Distance(const float3& a, const float3& b) 
{
	return Magnitude(Sub(b, a));
}

__device__ __forceinline__ float Dot(const float3& a, const float3& b) 
{
	return (a.x * b.x) + (a.y * b.y) + (a.z * b.z);
}

__device__ __forceinline__ float3 Cross(const float3& a, const float3& b) 
{
	return
	{
		(a.y * b.z) - (a.z * b.y),
		(a.z * b.x) - (a.x * b.z),
		(a.x * b.y) - (a.y * b.x)
	};
}

__device__ __forceinline__ float3 Normalize(const float3& f) 
{
	float magnitude = Magnitude(f);
	return Div(f, magnitude);
}

#endif