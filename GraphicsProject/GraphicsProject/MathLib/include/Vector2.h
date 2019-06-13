#pragma once
#include "MathLib.h"

namespace NVZMathLib 
{
	struct Vector2 
	{
		float x;
		float y;

		// Static template vectors:

		MATHLIBRARY_API Vector2();
		MATHLIBRARY_API Vector2(float new_x, float new_y);
		MATHLIBRARY_API Vector2(float new_all);
		MATHLIBRARY_API ~Vector2();

		// Operators

		MATHLIBRARY_API Vector2 operator + (const Vector2& other) const;

		MATHLIBRARY_API Vector2 operator - (const Vector2& other) const;

		MATHLIBRARY_API Vector2 operator * (const float& other) const;

		MATHLIBRARY_API Vector2 operator / (const float& other) const;

		MATHLIBRARY_API void operator += (const Vector2& other);

		MATHLIBRARY_API void operator -= (const Vector2& other);

		MATHLIBRARY_API void operator *= (const float& other);

		MATHLIBRARY_API void operator /= (const float& other);

		MATHLIBRARY_API Vector2 operator - () const;

		MATHLIBRARY_API float& operator[] (const int index);

		// Cast operators

		explicit MATHLIBRARY_API operator float*();

		// Other functions

		// Returns the magnitude of the vector.
		MATHLIBRARY_API float Magnitude() const;

		// Returns the square magnitiude of the vector.
		MATHLIBRARY_API float SqrMagnitude() const;

		// Returns an instance of the unit vector of this vector.
		MATHLIBRARY_API Vector2 Normalised() const;

		// Converts this vector to it's unit vector.
		MATHLIBRARY_API void Normalise();

		// Returns the normal of the vector.
		MATHLIBRARY_API Vector2 Normal() const;

		MATHLIBRARY_API float Dot(const Vector2& b) const;

		// Linear interpolation function.
		static MATHLIBRARY_API Vector2 Lerp(const Vector2& start, const Vector2& end, float time);

		// Static product functions
		
		static MATHLIBRARY_API float Dot(const Vector2& a, const Vector2& b);
	};

	static Vector2 operator* (float lhs, const Vector2& rhs)
	{
		return Vector2(rhs.x * lhs, rhs.y * lhs);
	}

	static Vector2 operator/ (float lhs, const Vector2& rhs)
	{
		return Vector2(rhs.x / lhs, rhs.y / lhs);
	}
}