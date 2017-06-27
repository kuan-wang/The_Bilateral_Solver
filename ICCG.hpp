#include <iostream>
#include <vector>
#include <math.h>

using namespace std;

/******************************
 * compressed sparse row format (csr)
********************************/
struct str_CSR
{
	double * val;
	int * 	col_index;
	int * 	row_index;
	int 	str_size;
	int 	row_size;
	int 	col_size;
};

struct str_CSR_colsort
{
	int 	num[7];
	int	 	row_index[7];
	int		size;
};

void make_CSRcolIndex(str_CSR * csr_mat_l, str_CSR_colsort * csr_col);
void executeIcdCsrFormat(str_CSR * csr_src , str_CSR * csr_dst, vector<double> &vec_d);
void ICResCsrFormat(str_CSR * csr_matl, str_CSR * csr_matl2, vector<double> vec_d, vector<double>  vec_r, vector<double> &vec_u);
void ApproximateSolution0(str_CSR * csr_mat, vector<double> vec_b, vector<double> vec_x, vector<double> &vec_r);
double dot(vector<double> vec1, vector<double> vec2, int n);
double dot_CSR(str_CSR * csr_mat, vector<double> &vec2, int row);
void transposition_Lmatrix(str_CSR * csr_mat, str_CSR * csr_mat2, int loop_cut);
int ICCGSolver(str_CSR * csr_mat, vector<double> vec_b, vector<double> &vec_x, int iter, double eps, str_CSR_colsort * csr_col);
double read_elementsCSR_skip(str_CSR * csr_mat, int i, int &j);
str_CSR_colsort * pre_ICD(str_CSR * csr_mat);

/*For Debug*/
double read_elementsCSR(str_CSR * csr_mat, int i, int j);
int rewrite_elementsCSR(str_CSR * csr_mat, double val, int i, int j);
int add_elementsCSR(str_CSR * csr_mat, double val, int i, int j);
void preview_CSR(str_CSR * csr);
void make_data(str_CSR * csr, int r_size);
void make_testData(str_CSR * csr);

/*Not use*/
void ICRes(str_CSR * csr_matl, vector<double> vec_d, vector<double>  vec_r, vector<double> &vec_u);
void IncompleteCholeskyDecomp(str_CSR * csr_src , str_CSR * csr_dst, vector<double> &vec_d);


