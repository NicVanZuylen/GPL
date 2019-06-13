#ifndef FLOAT3_FUNCS
#define FLOAT3_FUNCS

#ifndef __global__
#define __global__
#define __host__
#define __device__
#define __forceinline__

#define sqrtf;

struct float3 
{
	float x;
	float y;
	float z;
};

#endif

#define VECTOR_QUALIFIERS __device__ __forceinline__

struct vec3
{
	// -----------------------------------------------------------------------------------------------------
	// Structure

	float x;
	float y;
	float z;

	// -----------------------------------------------------------------------------------------------------
	// Constructors

	VECTOR_QUALIFIERS vec3(float new_x, float new_y, float new_z)
	{
		x = new_x;
		y = new_y;
		z = new_z;
	}

	VECTOR_QUALIFIERS vec3(float all)
	{
		x = all;
		y = all;
		z = all;
	}

	// -----------------------------------------------------------------------------------------------------
	// Operators

	VECTOR_QUALIFIERS vec3 operator + (const vec3& other) const
	{
		return vec3(x + other.x, y + other.y, z + other.z);
	}

	VECTOR_QUALIFIERS vec3 operator - (const vec3& other) const
	{
		return vec3(x - other.x, y - other.y, z - other.z);
	}

	VECTOR_QUALIFIERS vec3 operator * (const vec3& other) const
	{
		return vec3(x * other.x, y * other.y, z * other.z);
	}

	VECTOR_QUALIFIERS vec3 operator / (const vec3& other) const
	{
		return vec3(x / other.x, y / other.y, z / other.z);
	}

	VECTOR_QUALIFIERS vec3 operator * (const float& other) const
	{
		return vec3(x * other, y * other, z * other);
	}

	VECTOR_QUALIFIERS vec3 operator / (const float& other) const
	{
		return vec3(x / other, y / other, z / other);
	}

	VECTOR_QUALIFIERS void operator += (const vec3& other)
	{
		x += other.x;
		y += other.y;
		z += other.z;
	}

	VECTOR_QUALIFIERS void operator -= (const vec3& other)
	{
		x -= other.x;
		y -= other.y;
		z -= other.z;
	}

	VECTOR_QUALIFIERS void operator *= (const vec3& other)
	{
		x *= other.x;
		y *= other.y;
		z *= other.z;
	}

	VECTOR_QUALIFIERS void operator /= (const vec3& other)
	{
		x /= other.x;
		y /= other.y;
		z /= other.z;
	}

	VECTOR_QUALIFIERS void operator += (const float& other)
	{
		x += other;
		y += other;
		z += other;
	}

	VECTOR_QUALIFIERS void operator -= (const float& other)
	{
		x -= other;
		y -= other;
		z -= other;
	}

	VECTOR_QUALIFIERS void operator *= (const float& other)
	{
		x *= other;
		y *= other;
		z *= other;
	}

	VECTOR_QUALIFIERS void operator /= (const float& other)
	{
		x /= other;
		y /= other;
		z /= other;
	}

	// -----------------------------------------------------------------------------------------------------
	// Functions

	VECTOR_QUALIFIERS float Magnitude() const
	{
		return sqrtf((x * x) + (y * y) + (z * z));
	}

	VECTOR_QUALIFIERS float SqrMagnitude() const
	{
		return (x * x) + (y * y) + (z * z);
	}

	VECTOR_QUALIFIERS float Distance(const vec3& other) const
	{
		vec3 diff = other - vec3(x, y, z);

		return diff.Magnitude();
	}

	VECTOR_QUALIFIERS float Dot(const vec3& other) const
	{
		return (x * other.x) + (y * other.y) + (z * other.z);
	}

	VECTOR_QUALIFIERS vec3 Cross(const vec3& other) const
	{
		vec3 result
		(
			(y * other.z) - (z * other.y),
			(z * other.x) - (x * other.z),
			(x * other.y) - (y * other.x)
		);

		return result;
	}

	VECTOR_QUALIFIERS void Normalize()
	{
		float magnitude = Magnitude();
		
		x /= magnitude;
		y /= magnitude;
		z /= magnitude;
	}

	VECTOR_QUALIFIERS vec3 Normalized() const
	{
		float magnitude = Magnitude();

		return vec3(x / magnitude, y / magnitude, z / magnitude);
	}
};

VECTOR_QUALIFIERS float Dot(const vec3& a, const vec3& b)
{
	return (a.x * b.x) + (a.y * b.y) + (a.z * b.z);
}

VECTOR_QUALIFIERS vec3 Cross(const vec3& a, const vec3& b)
{
	vec3 result
	(
		(a.y * b.z) - (a.z * b.y),
		(a.z * b.x) - (a.x * b.z),
		(a.x * b.y) - (a.y * b.x)
	);

	return result;
}

VECTOR_QUALIFIERS float Magnitude(const vec3& a)
{
	return a.Magnitude();
}

VECTOR_QUALIFIERS vec3 Normalize(const vec3& a)
{
	return a.Normalized();
}

#endif