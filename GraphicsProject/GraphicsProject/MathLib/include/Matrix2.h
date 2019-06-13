#pragma once
#include "MathLib.h"
#include "Vector2.h"

namespace NVZMathLib 
{
	class Matrix2
	{
	public:

		MATHLIBRARY_API Matrix2();

		MATHLIBRARY_API Matrix2(float Xx, float Yx, float Xy, float Yy);

		MATHLIBRARY_API Matrix2(const Matrix2& other);

		MATHLIBRARY_API ~Matrix2();

		// Operators

		// Returns the float array which can be subscripted to access individual values.
		MATHLIBRARY_API Vector2& operator[] (const int index);

		// Concatenate matrix with another 2x2 matrix.
		MATHLIBRARY_API Matrix2 operator* (const Matrix2& other);

		// Return vector transformation of the input vector multiplied by this matrix.
		MATHLIBRARY_API Vector2 operator* (const Vector2& other);

		// Returns the identity matrix.
		static MATHLIBRARY_API Matrix2 Identity();

		// Cast operators

		// Returns the matrix casted to a float.
		explicit MATHLIBRARY_API operator float*();

		// Functions

		// Returns the determinant value of this matrix.
		MATHLIBRARY_API float Determinant();

		// Rotate the matrix anti-clockwise.
		MATHLIBRARY_API void SetRotate(float angle);

		// Gets the forward pointing vector of the matrix.
		MATHLIBRARY_API Vector2 Forward();

		// Gets the right pointing vector of the matrix.
		MATHLIBRARY_API Vector2 Right();

	private:

		// Initialize matrix to identity matrix.
		float m_nContents[2][2] =
		{
			{ 1.0f , 0.0f },
		    { 0.0f , 1.0f }
		};
	};
}
