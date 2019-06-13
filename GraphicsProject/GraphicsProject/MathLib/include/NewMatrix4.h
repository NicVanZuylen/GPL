#pragma once
#include "MathLib.h"
#include "Vector4.h"

namespace NVZMathLib 
{

	class NewMatrix4
	{
	public:

		MATHLIBRARY_API NewMatrix4();

		/*
		Matrix layout:

		[ x1 y1 z1 w1 ]
		[ x2 y2 z2 w2 ]
		[ x3 y3 z3 w3 ]
		[ x4 y4 z4 w4 ]
		*/
		MATHLIBRARY_API NewMatrix4
		(
			const float& x1, const float& x2, const float& x3, const float& x4,
			const float& y1, const float& y2, const float& y3, const float& y4,
			const float& z1, const float& z2, const float& z3, const float& z4,
			const float& w1, const float& w2, const float& w3, const float& w4
		);

		/*
		Matrix layout:

		[ x1 y1 z1 w1 ]
		[ x2 y2 z2 w2 ]
		[ x3 y3 z3 w3 ]
		[ x4 y4 z4 w4 ]
		*/
		MATHLIBRARY_API NewMatrix4(const Vector4& x, const Vector4& y, const Vector4& z, const Vector4& w);

		MATHLIBRARY_API ~NewMatrix4();

		MATHLIBRARY_API void ToIdentity();

		MATHLIBRARY_API void Transpose();

		MATHLIBRARY_API NewMatrix4 Transposed() const;

		MATHLIBRARY_API Vector4& operator [] (const int& index);

		explicit MATHLIBRARY_API operator float*();

		MATHLIBRARY_API NewMatrix4 operator * (const NewMatrix4& other);

		MATHLIBRARY_API Vector4 operator * (const Vector4& other);

		MATHLIBRARY_API void SetPosition(const float& x, const float& y, const float& z);
		MATHLIBRARY_API void SetPosition(const Vector4& v4Position);
		MATHLIBRARY_API void SetPosition(const Vector3& v3Position);

		MATHLIBRARY_API void Translate(const float& x, const float& y, const float& z);
		MATHLIBRARY_API void Translate(const Vector4& v4Translation);
		MATHLIBRARY_API void Translate(const Vector3& v3Translation);

		MATHLIBRARY_API void Rotate(const float& x, const float& y, const float& z);
		MATHLIBRARY_API void Rotate(const Vector4& eulerAngles);
		MATHLIBRARY_API void Rotate(const Vector3& eulerAngles);

		MATHLIBRARY_API void Scale(const float& x, const float& y, const float& z);
		MATHLIBRARY_API void Scale(const Vector4& v4ScaleMult);
		MATHLIBRARY_API void Scale(const Vector3& v3ScaleMult);

	private:

		// Column major layout:
		union
		{
			struct 
			{
				float x1;
				float x2;
				float x3;
				float x4;
				float y1;
				float y2;
				float y3;
				float y4;
				float z1; 
				float z2;
				float z3;
				float z4;
				float w1;
				float w2;
				float w3;
				float w4;
			};

			struct 
			{
				Vector4 m_xVec;
				Vector4 m_yVec;
				Vector4 m_zVec;
				Vector4 m_wVec;
			};

			Vector4 m_data[4];
		};
	};
}

