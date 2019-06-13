#pragma once
#include "MathLib.h"
#include "Vector3.h"
#include "Vector2.h"

namespace NVZMathLib
{
	class Matrix3
	{
	public:

		MATHLIBRARY_API Matrix3();

		MATHLIBRARY_API Matrix3(float Xx, float Yx, float Zx, float Xy, float Yy, float Zy, float Xz, float Yz, float Zz);

		MATHLIBRARY_API Matrix3(const Matrix3& other);

		MATHLIBRARY_API ~Matrix3();

		// Operators

		// Returns the float array which can be subscripted to access individual values.
		MATHLIBRARY_API Vector3& operator[] (const int index);

		// Concatenate matrix with another 3x3 matrix.
		MATHLIBRARY_API Matrix3 operator* (const Matrix3& other);

		// Return vector transformation of the input vector multiplied by this matrix.
		MATHLIBRARY_API Vector3 operator* (const Vector3& other);

		// Returns the identity matrix.
		static MATHLIBRARY_API Matrix3 Identity();

		// Cast operators

		// Returns the matrix casted to a float.
		explicit MATHLIBRARY_API operator float*();

		// Functions

		// Returns the determinant value of this matrix.
		MATHLIBRARY_API float Determinant();

		// Rotate on the X axis.
		MATHLIBRARY_API void SetRotateX(float angle);

		// Rotate on the Y axis.
		MATHLIBRARY_API void SetRotateY(float angle);

		// Rotate on the Z axis.
		MATHLIBRARY_API void SetRotateZ(float angle);

		// Sets the position point of the matrix using a 3D vector.
		MATHLIBRARY_API void SetPosition(Vector3 vPoint, bool bUseW = false);

		// Sets the position point of the matrix using a 2D vector.
		MATHLIBRARY_API void SetPosition(Vector2 vPoint);

		// Gets the position data stored within the matrix and returns it as a 2D vector.
		MATHLIBRARY_API Vector2 GetPosition();

		// Sets the scale vector of the matrix using a 3D vector.
		MATHLIBRARY_API void SetScale(Vector3 vScale, bool bUseW = false);

		// Sets the scale vector of the matrix using a 2D vector.
		MATHLIBRARY_API void SetScale(Vector2 vScale);

		// Gets the forward pointing vector of the matrix.
		MATHLIBRARY_API Vector2 Forward();

		// Gets the right pointing vector of the matrix.
		MATHLIBRARY_API Vector2 Right();

	private:

		// Initialize matrix to identity matrix.
		float m_nContents[3][3] =
		{
			{ 1.0f , 0.0f , 0.0f },
		    { 0.0f , 1.0f , 0.0f },
		    { 0.0f , 0.0f , 1.0f }
		};
	};
}
