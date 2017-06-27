#include <iostream>
#include <vector>
#include <math.h>
#include "ICCG.hpp"

using namespace std;

/*--------------------------------------------------------
func name : read_elementsCSR
note	  : x, y座標を指定してデータを読む。デバック用
--------------------------------------------------------*/
double read_elementsCSR(str_CSR * csr_mat, int i, int j)
{
	int x;
	int row, row_next;
	int col;
	double ret;

	row = csr_mat->row_index[i] - 1;
	row_next = csr_mat->row_index[i+1] - 1;
	for(x = row; x < row_next; x++)
	{
		col = csr_mat->col_index[x];
		if( col >= j )
			break;
	}

	if( col == j )
		ret = csr_mat->val[x];
	else
		ret = 0;

	return ret;
}


/*--------------------------------------------------------
func name : read_elementsCSR_skip
note	  : x, y座標を指定してデータを読む。
			使用しているが、あまりよい関数でない
--------------------------------------------------------*/
double read_elementsCSR_skip(str_CSR * csr_mat, int i, int &j)
{
	int x;
	int row;
	int row_next;
	int col;
	double ret;

	row = csr_mat->row_index[i] - 1;
	row_next = csr_mat->row_index[i+1] - 1;
	for(x = row; x < row_next; x++)
	{
		col = csr_mat->col_index[x];
		if( col >= j )
			break;
	}

	if(x == row_next)
	{
		ret = 0;
		j = csr_mat->col_size;
	}
	else
	{
		ret = csr_mat->val[x];
		j = col;
	}

	return ret;
}



/*--------------------------------------------------------
func name : rewrite_elementsCS
note	  : 要素の上書き。デバック用
--------------------------------------------------------*/
int rewrite_elementsCSR(str_CSR * csr_mat, double val, int i, int j)
{
	int x, row, loop_max, col;
	int ret = 0;

	row = csr_mat->row_index[i] - 1;
	loop_max = csr_mat->str_size;
	for(x = row; x < loop_max; x++)
	{
		col = csr_mat->col_index[x];
		if( col >= j )
			break;
	}

	if( col == j )
	{
		csr_mat->val[x] = val;
	}
	else
	{
		ret = 1;/*書き込めない場合*/
	}
	return ret;
}


/*--------------------------------------------------------
func name : add_elementsCSR
note	  : 要素の追加。デバック用
--------------------------------------------------------*/
int add_elementsCSR(str_CSR * csr_mat, double val, int i, int j)
{
	int x, row;
	int ret = 0;
	int flg_newrow = 0;
	int loop_max = csr_mat->str_size;

	//新しい行の追加判定
	if(csr_mat->row_index[i+1] == 0)
	{
		csr_mat->row_size = i+1;
		flg_newrow = 1;
		csr_mat->row_index[0] = 1;
	}

	//追加先のサーチ
	row = csr_mat->row_index[i] - 1;
	for(x = row; x < loop_max; x++)
	{
		if( csr_mat->val[x] == 0 )
			break;
	}

	//追加処理
	if( j > csr_mat->col_index[x-1] || flg_newrow == 1)
	{
		csr_mat->val[x] = val;
		csr_mat->col_index[x] = j;
		csr_mat->row_index[i+1] = x+2;
	}
	else
		ret = 1;	//追加できない場合

	return ret;
}


/*--------------------------------------------------------
func name : preview_CSR
note	  : 要素の一覧表示。デバック用
--------------------------------------------------------*/
void preview_CSR(str_CSR * csr)
{
	int i, j;
	double ret;

	for( i=0 ; i < csr->row_size ; i++)
	{
		for( j=0 ; j<csr->col_size ; j++)
		{
			ret = read_elementsCSR(csr, i, j);
			cout << ret << " ";
		}
		cout << endl;
	}
}

/*--------------------------------------------------------
func name : make_data
note	  : サンプルデータを作成する。デバック用
--------------------------------------------------------*/
void make_data(str_CSR * csr, int r_size)
{
	int i, j, k;
	int c_size = r_size * 3 - 2;
	csr->val = new double [c_size];
	csr->col_index = new int [c_size];
	csr->row_index = new int [r_size+1];
	csr->str_size = c_size;
	csr->row_size = r_size;
	csr->col_size = r_size;

	k = 0;
	csr->row_index[0] = 1;
	for( i=0 ; i<r_size; i++)
	{
		for( j=i-1 ; j<i+2; j++)
		{
			if(j < 0 || j >= r_size)
				continue;

			if( i == j)
				csr->val[k] = 2;
			else
				csr->val[k] = 1;
			csr->col_index[k] = j;
			k++;
		}
		csr->row_index[i+1] = k+1;
	}
}

/*--------------------------------------------------------
func name : executeIcdCsrFormat
note	  : 不完全コレスキー分解。
			CSRフォーマット用に高速で動作するように記述
--------------------------------------------------------*/
void executeIcdCsrFormat(str_CSR * csr_src , str_CSR * csr_dst, vector<double> &vec_d)
{
	int 		i, j ,k ,l;		/*loop変数*/
	int 		loop_k, loop_l;	/*k,lのloop回数を格納*/
	double 		lld;			/*L行列成分の途中計算を格納*/
	double * 	tmp_val;	/*L行列のサイズを特定できないので*/
	int * 		tmp_col;	/*出力データを一時的に格納*/
	double * 	src_val;	/*配列の要素ポインタ*/
	int * 		src_col;	/*(アロー演算子の記述が面倒なので)*/
	int * 		src_row;	/* 同様 */
	double * 	dst_val;	/* 同様 */
	int * 		dst_col;	/* 同様 */
	int * 		dst_row;	/* 同様 */
	int 		tmp_index=0;
	int 		j_index;

	/*メモリ確保 & 値設定*/
	csr_dst->row_index = new int [csr_src->row_size +1];
	// csr_dst->row_index = new int [csr_src->row_size];
	std::cout << "csr_src.row_size:" << csr_src->row_size << std::endl;

	csr_dst->col_size = csr_src->col_size;
	csr_dst->row_size = csr_src->row_size;
	csr_dst->str_size = csr_src->str_size;
	vec_d.resize(csr_src->row_size);
	// vec_d = new double [csr_src->row_size];

	/*ひとまず多めにメモリ確保*/
	tmp_val = new double [csr_src->str_size];
	tmp_col = new int [csr_src->str_size];

	/*ポインタのセット*/
	src_val = csr_src->val;
	src_col = csr_src->col_index;
	src_row = csr_src->row_index;
	dst_row = csr_dst->row_index;

	/*１行目のデータは自明*/
	tmp_val[0] = src_val[0];
	tmp_col[0] = src_col[0];
	vec_d[0] = 1.0 / tmp_val[0];

	/*行の開始情報は２行目まで自明*/
	dst_row[0] = 1;
	dst_row[1] = 2;

	std::cout << "csr_dst.row_size:" << csr_dst->row_size << std::endl;
	for(i = 1; i < csr_dst->row_size; i++){
		for(j = src_row[i]-1; j < src_row[i+1]-1; j++){

			if( i < src_col[j])
			{
				break; /*上三角成分は計算しない*/
			}

			lld = src_val[j];
			loop_k = j - src_row[i];
			j_index = src_col[j];	/*for分を見やすくするため*/

			for(k = dst_row[i]-1; k < dst_row[i]+loop_k; k++){
				loop_l = k - dst_row[i] + 1;
				for(l = dst_row[j_index]-1; l < dst_row[j_index]+loop_l ; l++){
					if(tmp_col[k] == tmp_col[l])
					{
						lld -= tmp_val[k] * tmp_val[l] * vec_d[tmp_col[l]];
					}
				}
			}

			tmp_index++;
			tmp_val[tmp_index] = lld;
			tmp_col[tmp_index] = src_col[j];
		}
		vec_d[i] = 1.0 / tmp_val[tmp_index];
		dst_row[i+1] = tmp_index+2;
	}
	// std::cout << __LINE__ << "delete 1" << std::endl;
	// delete tmp_val;
	// std::cout << __LINE__ << "delete 1" << std::endl;
	// delete tmp_col;
	// std::cout << __LINE__ << "delete 1" << std::endl;

	/*出力のためのメモリ確保 & ポインタ設定*/
	csr_dst->col_index = new int [tmp_index+1];
	csr_dst->val = new double [tmp_index+1];
	dst_val = csr_dst->val;
	dst_col = csr_dst->col_index;

	/*tmpのデータを移動*/
	for(i=0; i<tmp_index+1; i++)
	{
		dst_val[i] = tmp_val[i];
		dst_col[i] = tmp_col[i];
	}
	//TODO: there is a bug when run lena.jpg   solved:  csr_dst->row_index = new int [csr_src->row_size+1]
	delete tmp_val;
	std::cout << __LINE__ << "delete 1" << std::endl;
	delete tmp_col;
	std::cout << __LINE__ << "delete 1" << std::endl;
}


/*--------------------------------------------------------
func name : IncompleteCholeskyDecom
note	  : 不完全コレスキー分解。速度が遅いので未使用
--------------------------------------------------------*/
void IncompleteCholeskyDecomp(str_CSR * csr_src , str_CSR * csr_dst, vector<double> &vec_d)
{
	int 		i, j ,k;		/*loop変数*/
	double 		lld;			/*L行列成分の途中計算を格納*/
	double		tmp_val, l_ik, l_jk; /*一時計算用*/
	int 		str_size;
	int			element_count = 1;

   	vec_d.resize(csr_src->row_size);
	/*多めにメモリ確保するが気にしない*/
	csr_dst->val = new double [csr_src->str_size]();
	csr_dst->col_index = new int [csr_src->str_size]();
	csr_dst->row_index = new int [csr_src->row_size]();
	csr_dst->str_size = csr_src->str_size;
	csr_dst->col_size = csr_src->col_size;

	/*１行目のデータは自明*/
	tmp_val = read_elementsCSR(csr_src, 0, 0);
	add_elementsCSR(csr_dst, tmp_val, 0, 0);
	vec_d[0] = 1.0 / read_elementsCSR(csr_dst, 0 ,0);

	for(i = 1; i < csr_src->row_size; i++){
		for(j = 0; j <= i; j++){

			lld = read_elementsCSR(csr_src, i, j);
			if(lld == 0)
				continue;

			for(k = 0; k < j; k++)
			{
				l_ik = read_elementsCSR(csr_dst, i, k);
				l_jk = read_elementsCSR(csr_dst, j, k);
				lld -= l_ik * l_jk * vec_d[k];
			}
			add_elementsCSR(csr_dst, lld, i, j);
			element_count++;
		}
		vec_d[i] = 1.0 / read_elementsCSR(csr_dst, i, i);
	}
	csr_dst->str_size = element_count;
}

/*--------------------------------------------------------
func name : ICRes
note	  : 下三角行列による計算。遅いので未使用
--------------------------------------------------------*/
void ICRes(str_CSR * csr_matl, vector<double> vec_d, vector<double>  vec_r, vector<double> &vec_u)
{
	int i, j;
	double rly;
	double lu;
	vector <double> y(csr_matl->row_size);

	for( i = 0 ; i < csr_matl->row_size ; i++ )
	{
		rly = vec_r[i];
		for( j = 0 ; j < i ; j++ )
		{
			rly -=  y[j] * read_elementsCSR(csr_matl, i, j);
		}
		y[i] = rly/read_elementsCSR(csr_matl, i, i);
	}

	for( i = csr_matl->row_size-1; i >= 0; --i){
		lu = 0.0;
		for( j = i+1; j < csr_matl->row_size; ++j){
			lu += vec_u[j] * read_elementsCSR(csr_matl, j, i);
		}
		vec_u[i] = y[i]-vec_d[i]*lu;
	}
}

/*--------------------------------------------------------
func name : ICResCsrFormat
note	  : 下三角行列による計算。
　　　　　　CSRフォーマット用に高速化
--------------------------------------------------------*/
void ICResCsrFormat(str_CSR * csr_matl, str_CSR * csr_matl2, vector<double> vec_d, vector<double>  vec_r, vector<double> &vec_u)
{
	int i, j, k;
	int j_index;
	int i_index;
	double rly;
	double lu;
	vector <double> y(csr_matl->row_size);

	/*ポインタのセット*/
	double * 	src_val;	/*配列の要素ポインタ*/
	int * 		src_col;	/*(アロー演算子の記述が面倒なので)*/
	int * 		src_row;	/* 同様 */
	src_val = csr_matl->val;
	src_col = csr_matl->col_index;
	src_row = csr_matl->row_index;

	double * 	src_val2;	/*配列の要素ポインタ*/
	int * 		src_col2;	/*(アロー演算子の記述が面倒なので)*/
	int * 		src_row2;	/* 同様 */
	src_val2 = csr_matl2->val;
	src_col2 = csr_matl2->col_index;
	src_row2 = csr_matl2->row_index;

	for( i = 0 ; i < csr_matl->row_size ; i++ )
	{
		rly = vec_r[i];
		for(j = src_row[i]-1; j < src_row[i+1]-2; j++)
		{
			j_index = src_col[j];	/*for分を見やすくするため*/
			rly -=  y[j_index] * src_val[j];
		}
		y[i] = rly/src_val[j];
	}

	for( i = 0; i < csr_matl2->row_size ; i++)
	{
		lu = 0.0;
		for(j = src_row2[i]-1; j < src_row2[i+1]-2; j++)
		{
			i_index = csr_matl2->row_size - 1 - src_col2[j];	/*for分を見やすくするため*/
			lu +=  vec_u[i_index] * src_val2[j];
		}
		i_index = csr_matl2->row_size - 1 - i;	/*for分を見やすくするため*/
		vec_u[i_index] = y[i_index]-vec_d[i_index]*lu;
	}
}


/*--------------------------------------------------------
func name : make_CSRcolIndex
note	  : CSRフォーマットに対応した内積計算
--------------------------------------------------------*/
void make_CSRcolIndex(str_CSR * csr_mat_l, str_CSR_colsort * csr_col)
{
	int i, j;
	int cnt;

	double * 	src_val;	/*配列の要素ポインタ*/
	int * 		src_col;	/*(アロー演算子の記述が面倒なので)*/
	int * 		src_row;	/* 同様 */
	/*ポインタのセット*/
	src_val = csr_mat_l->val;
	src_col = csr_mat_l->col_index;
	src_row = csr_mat_l->row_index;

	for( i = 0 ; i < csr_mat_l->row_size; i++ )
	{
		csr_col[i].size = 0;
	}

	for( i = 0 ; i < csr_mat_l->row_size ; i++ )
	{
		for(j = src_row[i]-1; j < src_row[i+1]-1; j++)
		{
			cnt = csr_col[src_col[j]].size;
			csr_col[src_col[j]].num[cnt] = j;
			csr_col[src_col[j]].row_index[cnt] = csr_mat_l->row_size - i - 1;
			csr_col[src_col[j]].size++;
		}
	}

}


/*--------------------------------------------------------
func name : ApproximateSolution0
note	  : 第0近似解を求める。
--------------------------------------------------------*/
void ApproximateSolution0(str_CSR * csr_mat, vector<double> vec_b, vector<double> vec_x, vector<double> &vec_r)
{
	int i, j;
	int j_index;
	double ax;

    // 第0近似解に対する残差の計算
	for( i = 0 ; i < csr_mat->row_size ; i++ )
	{
		ax = 0.0;
		for(j = csr_mat->row_index[i]-1; j < csr_mat->row_index[i+1]-1; j++)
		{
			j_index = csr_mat->col_index[j];
			ax += vec_x[j_index] * csr_mat->val[j];
		}
		vec_r[i] = vec_b[i]-ax;
	}
}


/*--------------------------------------------------------
func name : dot
note	  : ベクトルの内積
--------------------------------------------------------*/
double dot(vector<double> vec1, vector<double> vec2, int n)
{
	double ret = 0;
	for(int i = 0; i < n; ++i){
		ret += vec1[i] * vec2[i];
	}

	return ret;
}


/*--------------------------------------------------------
func name : dot_CSR
note	  : CSRフォーマットに対応した内積計算
--------------------------------------------------------*/
double dot_CSR(str_CSR * csr_mat, vector<double> &vec2, int row)
{
	int row_s = csr_mat->row_index[row] - 1;
	int row_e = csr_mat->row_index[row+1] - 1;
	double ret = 0;
	for(int i = row_s; i < row_e; ++i){
		ret += csr_mat->val[i] * vec2[csr_mat->col_index[i]];
	}
	return ret;
}


/*--------------------------------------------------------
func name : transposition_Lmatrix
note	  : 行列Lを転置する。共役勾配法を高速化するために行う。
			本プログラムで一番重い処理。loopを工夫する必要あり。
--------------------------------------------------------*/
void transposition_Lmatrix(str_CSR * csr_mat, str_CSR_colsort * csr_col, str_CSR * csr_mat2)
{
	int i, j, k;
	int i2;
	int row_index;
	int col_index;
	int count = 0;

	double * 	src_val;	/*配列の要素ポインタ*/
	int * 		src_col;	/*(アロー演算子の記述が面倒なので)*/
	int * 		src_row;	/* 同様 */
	src_val = csr_mat->val;
	src_col = csr_mat->col_index;
	src_row = csr_mat->row_index;
	double * 	src_val2;	/*配列の要素ポインタ*/
	int * 		src_col2;	/*(アロー演算子の記述が面倒なので)*/
	int * 		src_row2;	/* 同様 */
	int			skip_cnt;
	int			skip_cnt2;

	csr_mat2->val = new double [csr_mat->str_size]();
	csr_mat2->col_index = new int [csr_mat->str_size]();
	csr_mat2->row_index = new int [csr_mat->row_size+1]();
	csr_mat2->str_size = csr_mat->str_size;
	csr_mat2->col_size = csr_mat->col_size;
	csr_mat2->row_size = csr_mat->row_size;

	src_val2 = csr_mat2->val;
	src_col2 = csr_mat2->col_index;
	src_row2 = csr_mat2->row_index;

	src_row2[0] = 1;
	for( i = csr_mat->row_size-1; i >= 0; --i){
		i2 = csr_mat->row_size - i;

		for( j = csr_col[i].size - 1; j > -1; --j)
		{
			src_val2[count] = csr_mat->val[csr_col[i].num[j]];
			src_col2[count] = csr_col[i].row_index[j];
			count++;
		}
		src_row2[i2] = count + 1;
	}
}


/*--------------------------------------------------------
func name : pre_ICD
note	  : 不完全コレスキー分解による前処理付き共役勾配法
--------------------------------------------------------*/
str_CSR_colsort * pre_ICD(str_CSR * csr_mat)
{
	int size = csr_mat->row_size;
	vector<double> vec_d(size);
	str_CSR  csr_l_mat;
	str_CSR_colsort * csr_col;

	executeIcdCsrFormat(csr_mat , &csr_l_mat , vec_d);
	csr_col = new str_CSR_colsort[csr_l_mat.str_size];
	make_CSRcolIndex(&csr_l_mat, csr_col);

	return csr_col;
}


/*--------------------------------------------------------
func name : ICCGSolver
note	  : 不完全コレスキー分解による前処理付き共役勾配法
--------------------------------------------------------*/
int ICCGSolver(str_CSR * csr_mat, vector<double> vec_b, vector<double> &vec_x, int iter, double eps, str_CSR_colsort * csr_col)
{
	int size = csr_mat->row_size;
   	vector<double> vec_p(size);
   	vector<double> vec_y(size);
	vector<double> vec_r(size);
   	vector<double> vec_r2(size);
	vector<double> vec_d(size);
	vec_x.assign(size, 0);

	str_CSR  csr_l_mat;
	str_CSR  csr_l_mat2;

	executeIcdCsrFormat(csr_mat , &csr_l_mat , vec_d);
	make_CSRcolIndex(&csr_l_mat, csr_col);
	transposition_Lmatrix(&csr_l_mat, csr_col, &csr_l_mat2);
	cout << "ICD_FIN" << endl;

	ApproximateSolution0(csr_mat, vec_b, vec_x, vec_r);
	ICResCsrFormat(&csr_l_mat, &csr_l_mat2, vec_d, vec_r, vec_p);

	double rr0 = dot(vec_r, vec_p, size);
	double rr1;
	double alpha, beta;

	double e = 0.0;
	int k;
	cout << "LOOP_START" << endl;
	for(k = 0; k < iter; ++k){
		//cout << "ICCG_loop:" << k << endl;
		for(int i = 0; i < size; ++i){
			vec_y[i] = dot_CSR(csr_mat, vec_p, i);
		}

		alpha = rr0/dot(vec_p, vec_y, size);

		for(int i = 0; i < size; ++i){
			vec_x[i] += alpha*vec_p[i];
			vec_r[i] -= alpha*vec_y[i];
		}

		ICResCsrFormat(&csr_l_mat, &csr_l_mat2, vec_d, vec_r, vec_r2);
		rr1 = dot(vec_r, vec_r2, size);

		e = sqrt(rr1);
		if(e < eps){
			k++;
			break;
		}

		beta = rr1/rr0;
		for(int i = 0; i < size; ++i){
			vec_p[i] = vec_r2[i] + beta * vec_p[i];
		}

		rr0 = rr1;
	}

	return 1;
}


/*--------------------------------------------------------
func name : make_testData
note	  : テストデータの生成。デバック用
--------------------------------------------------------*/
void make_testData(str_CSR * csr)
{
	csr->val = new double [17];
	csr->col_index = new int [17];
	csr->row_index = new int [6];
	csr->str_size = 17;
	csr->row_size = 5;
	csr->col_size = 5;
	csr->row_index[0] = 1;
	csr->row_index[1] = 4;
	csr->row_index[2] = 7;
	csr->row_index[3] = 11;
	csr->row_index[4] = 14;
	csr->row_index[5] = 18;
	csr->col_index[0] = 0;
	csr->col_index[1] = 1;
	csr->col_index[2] = 2;
	csr->col_index[3] = 0;
	csr->col_index[4] = 1;
	csr->col_index[5] = 4;
	csr->col_index[6] = 0;
	csr->col_index[7] = 2;
	csr->col_index[8] = 3;
	csr->col_index[9] = 4;
	csr->col_index[10] = 2;
	csr->col_index[11] = 3;
	csr->col_index[12] = 4;
	csr->col_index[13] = 1;
	csr->col_index[14] = 2;
	csr->col_index[15] = 3;
	csr->col_index[16] = 4;
	csr->val[0] = 2;
	csr->val[1] = 1;
	csr->val[2] = 1;
	csr->val[3] = 1;
	csr->val[4] = 2;
	csr->val[5] = 1;
	csr->val[6] = 1;
	csr->val[7] = 2;
	csr->val[8] = 1;
	csr->val[9] = 1;
	csr->val[10] = 1;
	csr->val[11] = 2;
	csr->val[12] = 1;
	csr->val[13] = 1;
	csr->val[14] = 1;
	csr->val[15] = 1;
	csr->val[16] = 2;
}
