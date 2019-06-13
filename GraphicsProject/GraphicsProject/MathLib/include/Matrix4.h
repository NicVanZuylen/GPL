#pragma once
#include "MathLib.h"
#include "Vector4.h"
#include "Vector3.h"

namespace NVZMathLib
{
	class Matrix4
	{
	public:

		MATHLIBRARY_API Matrix4();

		MATHLIBRARY_API Matrix4
		(
			float Xx, float Yx, float Zx, float Wx, 
			float Xy, float Yy, float Zy, float Wy, 
			float Xz, float Yz, float Zz, float Wz,
			float Xt, float Yt, float Zt, float Wt
		);

		MATHLIBRARY_API Matrix4(const Matrix4& other);

		MATHLIBRARY_API ~Matrix4();

		// Operators

		// Returns the float array which can be subscripted to access individual values.
		MATHLIBRARY_API Vector4& operator[] (const int index);

		// Concatenate matrix with another 4x4 matrix.
		MATHLIBRARY_API Matrix4 operator* (const Matrix4& other);

		// Return vector transformation of the input vector multiplied by this matrix.
		MATHLIBRARY_API Vector4 operator* (const Vector4& other);

		// Returns the identity matrix.
		static MATHLIBRARY_API Matrix4 Identity();

		// Cast operators

		// Returns the matrix casted to a float.
		explicit MATHLIBRARY_API operator float*();

		// Functions

		// Rotate on the X axis.
		MATHLIBRARY_API void SetRotateX(float angle);

		// Rotate on the Y axis.
		MATHLIBRARY_API void SetRotateY(float angle);

		// Rotate on the Z axis.
		MATHLIBRARY_API void SetRotateZ(float angle);

		// Sets the position point of the matrix using a 4D vector.
		MATHLIBRARY_API void SetPosition(Vector4 vPoint, bool bUseW = false);

		// Sets the position point of the matrix using a 3D vector.
		MATHLIBRARY_API void SetPosition(Vector3 vPoint);

		// Gets the position data stored within the matrix and returns it as a 3D vector.
		MATHLIBRARY_API Vector3 SetPosition();

		// Sets the scale vector of the matrix using a 4D vector.
		MATHLIBRARY_API void SetScale(Vector4 vScale, bool bUseW = false);

		// Sets the scale vector of the matrix using a 3D vector.
		MATHLIBRARY_API void SetScale(Vector3 vScale);

		// Gets the forward pointing vector of the matrix.
		MATHLIBRARY_API Vector3 GetForward();

		// Gets the right pointing vector of the matrix.
		MATHLIBRARY_API Vector3 GetRight();

		// Gets the right pointing vector of the matrix.
		MATHLIBRARY_API Vector3 GetUp();

	private:

		// Initialize matrix to identity matrix.
		float m_nContents[4][4] =
		{
			{ 1.0f , 0.0f , 0.0f , 0.0f },
		    { 0.0f , 1.0f , 0.0f , 0.0f },
		    { 0.0f , 0.0f , 1.0f , 0.0f },
		    { 0.0f , 0.0f , 0.0f , 1.0f }
		};
	};
}

