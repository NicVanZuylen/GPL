#pragma once
#include "MathLib.h"
#include "Vector3.h"
#include "Vector2.h"

namespace NVZMathLib
{
	struct Vector4
	{
		float x;
		float y;
		float z;
		float w;

		// Static template vectors:

		MATHLIBRARY_API Vector4();
		MATHLIBRARY_API Vector4(float new_x, float new_y, float new_z, float new_w);
		MATHLIBRARY_API Vector4(float new_all);
		MATHLIBRARY_API Vector4(const Vector3& new_xyz, float w);
		MATHLIBRARY_API Vector4(float new_x, const Vector3& new_yzw);
		MATHLIBRARY_API Vector4(const Vector2& new_xy, const Vector2& new_zw);
		MATHLIBRARY_API Vector4(const Vector2& new_repeatXY);
		MATHLIBRARY_API ~Vector4();

		// Swizzling

		MATHLIBRARY_API Vector3 XYZ() const;
		MATHLIBRARY_API Vector3 ZYX() const;
		MATHLIBRARY_API Vector3 YXZ() const;
		MATHLIBRARY_API Vector3 ZXY() const;
		MATHLIBRARY_API Vector2 XY() const;
		MATHLIBRARY_API Vector2 YZ() const;
		MATHLIBRARY_API Vector2 XZ() const;

		// Operators

		MATHLIBRARY_API Vector4 operator + (const Vector4& other) const;

		MATHLIBRARY_API Vector4 operator - (const Vector4& other) const;

		MATHLIBRARY_API Vector4 operator * (const float& other) const;

		MATHLIBRARY_API Vector4 operator / (const float& other) const;

		MATHLIBRARY_API void operator += (const Vector4& other);

		MATHLIBRARY_API void operator -= (const Vector4& other);

		MATHLIBRARY_API void operator *= (const float& other);

		MATHLIBRARY_API void operator /= (const float& other);

		MATHLIBRARY_API Vector4 operator - () const;

		MATHLIBRARY_API float& operator[] (const int index);

		// Cast operators

		explicit MATHLIBRARY_API operator float*();

		// Other functions

		// Returns the magnitude of the vector.
		MATHLIBRARY_API float Magnitude() const;

		// Returns an instance of the unit vector of this vector.
		MATHLIBRARY_API Vector4 Normalised() const;

		// Converts this vector to it's unit vector.
		MATHLIBRARY_API void Normalise();

		MATHLIBRARY_API float Dot(const Vector4& b) const;

		MATHLIBRARY_API Vector4 Cross(const Vector4& b) const;

		static MATHLIBRARY_API Vector4 Lerp(const Vector4& start, const Vector4& end, float time);

		// Static product functions

		static MATHLIBRARY_API float Dot(const Vector4& a, const Vector4& b);

		static MATHLIBRARY_API Vector4 Cross(const Vector4& a, const Vector4& b);
	};

	static Vector4 operator* (const float& lhs, const Vector4& rhs)
	{
		return Vector4(rhs.x * lhs, rhs.y * lhs, rhs.z * lhs, 0.0f);
	}

	static Vector4 operator/ (const float& lhs, const Vector4& rhs)
	{
		return Vector4(rhs.x / lhs, rhs.y / lhs, rhs.z * lhs, 0.0f);
	}
}
