#include "GPL_VertexAttributes.h"
#include "GLAD/glad.h"

using namespace GPL;

GPL_VertexAttributes::GPL_VertexAttributes() 
{

}

GPL_VertexAttributes::~GPL_VertexAttributes() 
{

}

void GPL_VertexAttributes::AddAttribute(EAttributeType type, unsigned char divisor) 
{
	m_attributes.push_back(type);
	m_divisors.push_back(divisor);
}

void GPL_VertexAttributes::AddAttribute(EAttributeType type, unsigned char divisor, int index)
{
	m_attributes.insert(m_attributes.begin() + index, type);
	m_divisors.insert(m_divisors.begin() + index, divisor);
}

void GPL_VertexAttributes::RemoveAttribute(int index)
{
	m_attributes.erase(m_attributes.begin() + index);
	m_divisors.erase(m_divisors.begin() + index);
}

void GPL_VertexAttributes::UseAttributes()
{
	// Assuming the VBO and VAO are already bound externally...
	int size = 0; // Used as the vertex stride value.
	for (int i = 0; i < m_attributes.size(); ++i) 
	{
		switch (m_attributes[i])
		{
		case GPL_FLOAT:
			size += sizeof(float);
			break;

		case GPL_FLOAT2:
			size += 2 * sizeof(float);
			break;

		case GPL_FLOAT3:
			size += 3 * sizeof(float);
			break;

		case GPL_FLOAT4:
			size += 4 * sizeof(float);
			break;

		case GPL_INT:
			size += sizeof(int);
			break;

		case GPL_INT2:
			size += sizeof(int) * 2;
			break;

		case GPL_INT3:
			size += sizeof(int) * 3;
			break;

		case GPL_INT4:
			size += sizeof(int) * 4;
			break;

		case GPL_SHORT:
			size += sizeof(short);
			break;

		case GPL_SHORT2:
			size += sizeof(short) * 2;
			break;

		case GPL_SHORT3:
			size += sizeof(short) * 3;
			break;

		case GPL_SHORT4:
			size += sizeof(short) * 4;
			break;

		case GPL_BYTE:
			size += 1;
			break;

		case GPL_BYTE2:
			size += 2;
			break;

		case GPL_BYTE3:
			size += 3;
			break;

		case GPL_BYTE4:
			size += 4;
			break;
		}
	}

	// Track current offset into the vertex, used for pointer values.
	unsigned long long currentOffset = 0;

	for (int i = 0; i < m_attributes.size(); ++i) 
	{
		switch (m_attributes[i])
		{
		case GPL_FLOAT:
			glVertexAttribPointer(i, 1, GL_FLOAT, GL_FALSE, size, (void*)currentOffset);
			currentOffset += sizeof(float);
			break;

		case GPL_FLOAT2:
			glVertexAttribPointer(i, 2, GL_FLOAT, GL_FALSE, size, (void*)currentOffset);
			currentOffset += sizeof(float) * 2;
			break;

		case GPL_FLOAT3:
			glVertexAttribPointer(i, 3, GL_FLOAT, GL_FALSE, size, (void*)currentOffset);
			currentOffset += sizeof(float) * 3;
			break;

		case GPL_FLOAT4:
			glVertexAttribPointer(i, 4, GL_FLOAT, GL_FALSE, size, (void*)currentOffset);
			currentOffset += sizeof(float) * 4;
			break;

		case GPL_INT:
			glVertexAttribPointer(i, 1, GL_INT, GL_FALSE, size, (void*)currentOffset);
			currentOffset += sizeof(int);
			break;

		case GPL_INT2:
			glVertexAttribPointer(i, 2, GL_INT, GL_FALSE, size, (void*)currentOffset);
			currentOffset += sizeof(int) * 2;
			break;

		case GPL_INT3:
			glVertexAttribPointer(i, 3, GL_INT, GL_FALSE, size, (void*)currentOffset);
			currentOffset += sizeof(int) * 3;
			break;

		case GPL_INT4:
			glVertexAttribPointer(i, 4, GL_INT, GL_FALSE, size, (void*)currentOffset);
			currentOffset += sizeof(int) * 4;
			break;

		case GPL_SHORT:
			glVertexAttribPointer(i, 1, GL_SHORT, GL_FALSE, size, (void*)currentOffset);
			currentOffset += sizeof(short);
			break;

		case GPL_SHORT2:
			glVertexAttribPointer(i, 2, GL_SHORT, GL_FALSE, size, (void*)currentOffset);
			currentOffset += sizeof(short) * 2;
			break;

		case GPL_SHORT3:
			glVertexAttribPointer(i, 3, GL_SHORT, GL_FALSE, size, (void*)currentOffset);
			currentOffset += sizeof(short) * 3;
			break;

		case GPL_SHORT4:
			glVertexAttribPointer(i, 4, GL_SHORT, GL_FALSE, size, (void*)currentOffset);
			currentOffset += sizeof(short) * 4;
			break;

		case GPL_BYTE:
			glVertexAttribPointer(i, 1, GL_BYTE, GL_FALSE, size, (void*)currentOffset);
			currentOffset += 1;
			break;

		case GPL_BYTE2:
			glVertexAttribPointer(i, 2, GL_BYTE, GL_FALSE, size, (void*)currentOffset);
			currentOffset += 2;
			break;

		case GPL_BYTE3:
			glVertexAttribPointer(i, 3, GL_BYTE, GL_FALSE, size, (void*)currentOffset);
			currentOffset += 3;
			break;

		case GPL_BYTE4:
			glVertexAttribPointer(i, 4, GL_BYTE, GL_FALSE, size, (void*)currentOffset);
			currentOffset += 4;
			break;
		}

		// Enable.
		glEnableVertexAttribArray(i);

		// Divisor
		glVertexAttribDivisor(i, m_divisors[i]);
	}
}

int GPL_VertexAttributes::AttributeCount() 
{
	return static_cast<int>(m_attributes.size());
}