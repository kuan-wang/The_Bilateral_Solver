/**
 * This file is part of the SparseMatrix library
 *
 * @license  MIT
 * @author   Petr Kessler (https://kesspess.cz)
 * @link     https://github.com/uestla/Sparse-Matrix
 */

#ifndef __SPARSEMATRIX_H__

	#define	__SPARSEMATRIX_H__

	#include <vector>
	#include <iostream>

	using namespace std;


	template<typename T>
	class SparseMatrix
	{

		public:

			// === CREATION ==============================================

			SparseMatrix(int n); // square matrix n×n
			SparseMatrix(int rows, int columns); // general matrix

			SparseMatrix(const SparseMatrix<T> & m); // copy constructor
			SparseMatrix<T> & operator = (const SparseMatrix<T> & m);

			~SparseMatrix(void);


			// === GETTERS / SETTERS ==============================================

			int getRowCount(void) const;
			int getColumnCount(void) const;


			// === VALUES ==============================================

			T get(int row, int col) const;
			SparseMatrix & set(T val, int row, int col);


			// === OPERATIONS ==============================================

			SparseMatrix<T> multiply(const T & x) const;
			SparseMatrix<T> operator * (const T & x) const;

			vector<T> multiply(const vector<T> & x) const;
			vector<T> operator * (const vector<T> & x) const;

			SparseMatrix<T> multiply(const SparseMatrix<T> & m) const;
			SparseMatrix<T> operator * (const SparseMatrix<T> & m) const;

			SparseMatrix<T> add(const SparseMatrix<T> & m) const;
			SparseMatrix<T> operator + (const SparseMatrix<T> & m) const;

			SparseMatrix<T> subtract(const SparseMatrix<T> & m) const;
			SparseMatrix<T> operator - (const SparseMatrix<T> & m) const;

			SparseMatrix<T> transpose(const SparseMatrix<T> & m) const;

			// === FRIEND FUNCTIONS =========================================

			template<typename X>
			friend bool operator == (const SparseMatrix<X> & a, const SparseMatrix<X> & b);

			template<typename X>
			friend bool operator != (const SparseMatrix<X> & a, const SparseMatrix<X> & b);

			template<typename X>
			friend ostream & operator << (ostream & os, const SparseMatrix<X> & matrix);


		protected:

			int m, n;

			vector<T> * vals;
			vector<int> * rows, * cols;


			// === HELPERS / VALIDATORS ==============================================

			void construct(int m, int n);
			void destruct(void);
			void deepCopy(const SparseMatrix<T> & m);
			void validateCoordinates(int row, int col) const;
			void insert(int index, int row, int col, T val);
			void remove(int index, int row);

	};

#endif
