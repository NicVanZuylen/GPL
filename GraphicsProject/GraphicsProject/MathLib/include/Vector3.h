#pragma once
#include "MathLib.h"
#include "Vector2.h"

namespace NVZMathLib
{
	struct Vector3
	{
		float x;
		float y;
		float z;

		// Static template vectors:

		MATHLIBRARY_API Vector3();
		MATHLIBRARY_API Vector3(float new_x, float new_y, float new_z);
		MATHLIBRARY_API Vector3(float new_all);
		MATHLIBRARY_API Vector3(const Vector2& new_xy, float new_z);
		MATHLIBRARY_API Vector3(float new_x, const Vector2& new_yz);
		MATHLIBRARY_API ~Vector3();

		// Swizzling

		MATHLIBRARY_API Vector2 XY() const;
		MATHLIBRARY_API Vector2 YX() const;
		MATHLIBRARY_API Vector2 YZ() const;
		MATHLIBRARY_API Vector2 XZ() const;
		MATHLIBRARY_API Vector2 ZX() const;
		MATHLIBRARY_API Vector2 ZY() const;

		// Operators

		MATHLIBRARY_API Vector3 operator + (const Vector3& other) const;

		MATHLIBRARY_API Vector3 operator - (const Vector3& other) const;

		MATHLIBRARY_API Vector3 operator * (const float& other) const;

		MATHLIBRARY_API Vector3 operator / (const float& other) const;

		MATHLIBRARY_API void operator += (const Vector3& other);

		MATHLIBRARY_API void operator -= (const Vector3& other);

		MATHLIBRARY_API void operator *= (const float& other);

		MATHLIBRARY_API void operator /= (const float& other);

		MATHLIBRARY_API Vector3 operator - () const;

		MATHLIBRARY_API float& operator[] (const int index);

		// Cast operators

		explicit MATHLIBRARY_API operator float*();

		// Other functions

		// Returns the magnitude of the vector.
		MATHLIBRARY_API float Magnitude() const;

		// Returns an instance of the unit vector of this vector.
		MATHLIBRARY_API Vector3 Normalised() const;

		// Converts this vector to it's unit vector.
		MATHLIBRARY_API void Normalise();

		MATHLIBRARY_API float Dot(const Vector3& b) const;

		MATHLIBRARY_API Vector3 Cross(const Vector3& b) const;

		static MATHLIBRARY_API Vector3 Lerp(const Vector3& start, const Vector3& end, float time);

		// Static product functions

		static MATHLIBRARY_API float Dot(const Vector3& a, const Vector3& b);

		static MATHLIBRARY_API Vector3 Cross(const Vector3& a, const Vector3& b);
	};

	static Vector3 operator* (const float& lhs, const Vector3& rhs)
	{
		return Vector3(rhs.x * lhs, rhs.y * lhs, rhs.z * lhs);
	}

	static Vector3 operator/ (const float& lhs, const Vector3& rhs)
	{
		return Vector3(rhs.x / lhs, rhs.y / lhs, rhs.z * lhs);
	}
}