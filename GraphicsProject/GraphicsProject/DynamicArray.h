#pragma once
#include <iostream>

template <typename T>
class DynamicArray
{
public:

	// Default expansion rate is 10.
	DynamicArray()
	{
		m_nExpandRate = 1;
		m_nSize = 1;
		m_nCount = 0;
		m_contents = new T[m_nSize];
	}

	DynamicArray(int nSize, int nExpandRate)
	{
		// Ensure start size and expand rate are never below 1.
		if (nSize < 1)
			nSize = 1;

		if (nExpandRate < 1)
			nExpandRate = 1;

		m_nExpandRate = nExpandRate;
		m_nSize = nSize;
		m_nCount = 0;
		m_contents = new T[m_nSize];
	}

	~DynamicArray()
	{
		if (m_contents != nullptr)
			delete[] m_contents;
	}

	// Getters
	const int GetExpandRate() const
	{
		return m_nExpandRate;
	}

	const int GetSize() const
	{
		return m_nSize;
	}

	const int Count() const
	{
		return m_nCount;
	}

	// Setters
	void SetExpandRate(int nExpandRate)
	{
		m_nExpandRate = nExpandRate;
	}

	// Push (add)

	// Adds a value to the end of the array, and expands the array if there is no room for the new value.
	void Push(const T& value)
	{
		if (m_nCount < m_nSize)
		{
			m_contents[m_nCount] = value;
		}
		else
		{
			Expand(value);
		}

		++m_nCount;
	}

	// Pop (remove)

	// Removes the value in the array at the specified index. The location of the removed value is replaced by its successor the junk value is moved to the end of the array.
	void PopAt(int index)
	{
		if(index < m_nSize - 1) 
		{
			int nCopySize = (m_nSize - (index + 1)) * sizeof(T);
			memcpy_s(&m_contents[index], nCopySize, &m_contents[index + 1], nCopySize);
		}

		// Decrease used slot count.
		--m_nCount;
	}

	// Decreases the array size to fit the amount of elements used to reduce RAM usage. (Slow)
	void ShrinkToFit()
	{
		int iShrinkBy = m_nExpandRate / m_nCount;

		m_nSize -= iShrinkBy * m_nExpandRate;

		// Temporary pointer to the new array.
		T* tmpContents = new T[m_nSize];

		// Copy old contents to new content array.
		memcpy_s(tmpContents, sizeof(T) * m_nSize, m_contents, sizeof(T) * m_nSize);

		// m_contents is no longer useful, delete it.

		delete[] m_contents;

		// Then set m_contents to the address of tmpContents

		m_contents = tmpContents;
	}

	T& operator [] (const int& index)
	{
		return m_contents[index];
	}

	const T& operator [] (const int& index) const
	{
		return m_contents[index];
	}

	void Clear()
	{
		m_nCount = 0;
	}

	const T* Data() 
	{
		return m_contents;
	}

	typedef bool(*CompFunctionPtr)(T lhs, T rhs);

	// Quick sort function. Takes in a function pointer used for sorting.
	void QuickSort(int nStart, int nEnd, CompFunctionPtr sortFunc)
	{
		if (nStart < nEnd) // Is false when finished.
		{
			int nPartitionIndex = Partition(nStart, nEnd, sortFunc); // The splitting point between sub-arrays/partitions.

			QuickSort(nStart, nPartitionIndex, sortFunc);
			QuickSort(nPartitionIndex + 1, nEnd, sortFunc);

			// Process repeats until the entire array is sorted.
		}
	}

private:

	// Partition function for quick sort algorithm.
	int Partition(int nStart, int nEnd, CompFunctionPtr sortFunc)
	{
		T nPivot = operator[](nEnd - 1);

		int smallPartition = nStart - 1; // AKA: i or the left partition slot.

		for (int j = smallPartition + 1; j < nEnd; ++j)
		{
			if (sortFunc(operator[](j), nPivot))
			{
				// Move selected left partition (i) slot.
				++smallPartition;

				// Move to left partition

				T tmp = operator[](smallPartition);

				operator[](smallPartition) = operator[](j);

				operator[](j) = tmp;
			}
		}
		// Swap next i and the pivot
		if (smallPartition < nEnd - 1)
		{
			T tmp = operator[](smallPartition + 1);

			operator[](smallPartition + 1) = operator[](nEnd - 1);

			operator[](nEnd - 1) = tmp;
		}

		return smallPartition;
	}

	// Expand function.
	void Expand(T overflowValue)
	{
		// Temporary pointer to the new array.
		T* tmpContents = new T[m_nSize + m_nExpandRate];

		// Copy old contents to new content array.
		memcpy_s(tmpContents, sizeof(T) * (m_nSize + m_nExpandRate), m_contents, sizeof(T) * m_nSize);

		// Add new value.
		tmpContents[m_nSize] = overflowValue;

		// m_contents is no longer useful, delete it.

		delete[] m_contents;

		// Then set m_contents to the address of tmpContents

		m_contents = tmpContents;

		// Change array size indicator.
		m_nSize += m_nExpandRate;
	}

	T* m_contents = nullptr;
	int m_nExpandRate;
	int m_nSize;
	int m_nCount;
};