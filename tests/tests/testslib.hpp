/**
 * This file is part of the SparseMatrix library
 *
 * @license  MIT
 * @author   Petr Kessler (https://kesspess.cz)
 * @link     https://github.com/uestla/Sparse-Matrix
 */

// #ifndef npixels
// #define npixels 10
// #endif
//
// #ifndef nvertices
// #define nvertices 10
// #endif
//
// #ifndef dim
// #define dim 5
// #endif




#ifndef __TESTSLIB_H__

	#define	__TESTSLIB_H__

	// #include<opencv2/core/core.hpp>
	// #include<opencv2/core/eigen.hpp>
	// #include<opencv2/highgui.hpp>
	// #include<opencv2/opencv.hpp>


	#include <time.h>
	#include <string>
	#include <vector>
	#include <cstring>
	#include <sstream>
	#include <iostream>
	#include <typeinfo>
	#include <exception>
    #include <unordered_map>
    #include <unordered_set>


	#include <Eigen/Dense>
	#include <Eigen/SparseCore>
	#include <Eigen/SparseCholesky>
	#include <Eigen/IterativeLinearSolvers>
	#include <Eigen/Sparse>

	// #include "omp.h"
	#include <omp.h>


	using namespace std;


static int npixels = 300;
static int nvertices = 10;
static int dim = 5;
static int pd = 5;
static int vd = 4;
static std::vector<Eigen::SparseMatrix<double> > blurs;
static std::vector<Eigen::Triplet<double> > triple_blur;
static std::vector<Eigen::Triplet<double> > triple_S;
static Eigen::SparseMatrix<double> blurs_test;
static Eigen::SparseMatrix<double> S;
static Eigen::SparseMatrix<double> Dn;
static Eigen::SparseMatrix<double> Dm;

    struct grid_params
    {
        double spatialSigma;
        double lumaSigma;
        double chromaSigma;
        grid_params()
        {
            spatialSigma = 8.0;
            lumaSigma = 4.0;
            chromaSigma = 4.0;
        }
    };

    struct bs_params
    {
        double lam;
        double A_diag_min;
        double cg_tol;
        int cg_maxiter;
        bs_params()
        {
            lam = 128.0;
            A_diag_min = 1e-5;
            cg_tol = 1e-5;
            cg_maxiter = 25;
        }
    };

static grid_params grid_param;
static bs_params bs_param;

const char** args;


	class FailureException : public exception
	{

		public:

			FailureException(const string & message) : exception(), message(message)
			{}


			virtual ~FailureException(void) throw ()
			{}


			inline string getMessage(void) const
			{
				return this->message;
			}


		protected:

			string message;

	};


	void assertException(const char * exceptionClass, void (*callback)(void))
	{
		try {
			callback();

		} catch (const exception & e) {
			string actualClass(typeid(e).name());

			if (strstr(actualClass.c_str(), exceptionClass) == NULL) {
				ostringstream oss;
				oss << "Exception class '" << exceptionClass << "' expected, but '" << actualClass << "' thrown.";

				throw FailureException(oss.str());
			}

			return ;
		}

		throw FailureException("Exception expected but none thrown.");
	}


	template<typename T>
	void assertEquals(const T & a, const T & b, const char * message = NULL)
	{
		if (!(a == b)) {
			ostringstream oss;
			if (message == NULL) {
				oss << "Objects not equal when they should be." << endl;

			} else {
				oss << message << endl;
			}

			oss << a << endl << "expected, but" << endl << b << " given";
			throw FailureException(oss.str());
		}
	}


	template<typename X, typename Y>
	void assertEquals(const X & a, const Y & b, const char * message = NULL)
	{
		if (!(a == b)) {
			ostringstream oss;
			if (message == NULL) {
				oss << "Objects not equal when they should be." << endl;

			} else {
				oss << message << endl;
			}

			oss << a << endl << "expected, but" << endl << b << " given";
			throw FailureException(oss.str());
		}
	}


	template<typename T>
	vector<T> generateRandomVector(int size)
	{
		vector<T> vector(size, 0);

		for (int i = 0; i < size; i++) {
			vector[i] = rand() % (2);
			// vector[i] = rand() % 101;
		}

		return vector;
	}

    template<typename T>
    void PrintVector(vector<T>& v)
    {
        for (typename vector<T>::const_iterator it = v.begin(); it != v.end(); it++) {
            cout<<*it<<" ";
        }
        cout<<endl;
    }

    template<typename T>
    void PrintUnordered_set(unordered_set<T>& s)
    {
        for (typename unordered_set<T>::const_iterator it = s.begin(); it != s.end(); it++) {
            cout<<*it<<" ";
        }
        cout<<endl;
    }



	template<typename T>
	vector<vector<T> > generateRandomMatrix(int rows, int columns)
	{
		vector<vector<T> > matrix(rows, vector<int>(columns, 0));

		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				matrix[i][j] = rand();
				// matrix[i][j] = rand() % 101;
			}
		}

		return matrix;
	}


	// === OUTPUT HELPERS =========================================

	template<typename T>
	ostream & operator << (ostream & os, const vector<T> & v)
	{
		os << "[";

		for (int i = 0, len = v.size(); i < len; i++) {
			if (i != 0) {
				os << ", ";
			}

			os << v[i];
		}

		os << "]";

		return os;
	}

#endif
