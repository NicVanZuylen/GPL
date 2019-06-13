#pragma once
#include <vector>

namespace GPL 
{
	enum EAttributeType 
	{
		GPL_FLOAT,
		GPL_FLOAT2,
		GPL_FLOAT3,
		GPL_FLOAT4,
		GPL_INT,
		GPL_INT2,
		GPL_INT3,
		GPL_INT4,
		GPL_SHORT,
		GPL_SHORT2,
		GPL_SHORT3,
		GPL_SHORT4,
		GPL_BYTE,
		GPL_BYTE2,
		GPL_BYTE3,
		GPL_BYTE4,
	};

	class GPL_VertexAttributes 
	{
	public:

		GPL_VertexAttributes();

		~GPL_VertexAttributes();

		/*
		Description: Add a vertex attribute to this attribute object.
		Param:
		    EAttributeType type: The data type of the attribute to add.
		*/
		void AddAttribute(EAttributeType type, unsigned char divisor);

		/*
		Description: Add a vertex attribute to this attribute object at the specified index.
		Param:
			EAttributeType type: The data type of the attribute to add.
			int index: The index to insert the new attribute at.
		*/
		void AddAttribute(EAttributeType type, unsigned char divisor, int index);

		/*
		Description: Remove a vertex attribute from this attribute object at the specified index.
		Param:
			int index: The index to remove the attribute from.
		*/
		void RemoveAttribute(int index);

		/*
		Description: Assign these vertex attributes to a currently bound VBO in the currently bound VAO.
		*/
		void UseAttributes();

		/*
		Description: Returns the amount of active attributes present.
		Return Type: int
		*/
		int AttributeCount();

	private:

		std::vector<EAttributeType> m_attributes;
		std::vector<unsigned char> m_divisors;
	};
};