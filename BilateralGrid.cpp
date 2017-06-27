
// #define _DEBUG_


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include "BilateralGrid.hpp"

using namespace cv;
using namespace std;

/****************************************************
brief	: Bilateral Gridクラスコンストラクタ
note	:
*****************************************************/
BilateralGrid::BilateralGrid(Mat3f mat_image)
{
	mat_input = get_Ych(mat_image);
	mat_output = mat_input.clone();
}

/****************************************************
brief	: 色を塗った画像をセットする
note	:
*****************************************************/
void BilateralGrid::set_DrawImage(Mat3f mat_draw_image)
{
	mat_color = mat_draw_image.clone();
	mat_colorized = mat_draw_image.clone();
}

void BilateralGrid::set_DepthImage(Mat1f mat_depth_image)
{
	mat_depth = mat_depth_image.clone();
	mat_depthsuperresoluted = mat_depth_image.clone();
}


/****************************************************
brief	: 行列Aを作成する。共通事項
note	:
*****************************************************/
void BilateralGrid::construct_AMatrix_step1()
{
	int i, j;
	float *check_vecter;
	st_blur * p_blur;
	st_A * p_amat;
	st_A * p_amat_U;
	st_A * p_amat_V;
	st_A * p_amat_D;

	b_vecter_U = new float [bg_size]();
	b_vecter_V = new float [bg_size]();
	b_vecter_D = new float [bg_size]();
	A_matrix = new st_A [bg_size];
	A_matrix_U = new st_A [bg_size];
	A_matrix_V = new st_A [bg_size];
	A_matrix_D = new st_A [bg_size];

	p_blur = blur_matrix;
	p_amat_U = A_matrix;

	/*calc NAN*/
	for(i=0; i<bg_size; i++)
	{
		p_amat_U->count = p_blur->count;

		p_amat_U->index[0] = p_blur->index[0];
		p_amat_U->value[0] = WEIGHT_CENTER * diagN_matrix[i] * diagN_matrix[i];
		for(j=1; j<p_amat_U->count; j++)
		{
			p_amat_U->index[j] = p_blur->index[j];
			/*重み付けは1*/
			p_amat_U->value[j] = -WEIGHT_NEIGHBOR * diagN_matrix[i] * diagN_matrix[p_blur->index[j]];
		}
		p_blur++;
		p_amat_U++;
	}

	/*D - NAN*/
	p_amat_U = A_matrix;
	for(i=0; i<bg_size; i++)
	{
		p_amat_U->value[0] = diagM_matrix[i] - p_amat_U->value[0];

		//共役勾配法対策、これがないとエラーになる
		if(p_amat_U->count ==1)
			p_amat_U->value[0] = 1;

		p_amat_U++;
	}

	/*ここまではU成分とY成分の計算行列は同じ*/
	p_amat = A_matrix;
	p_amat_U = A_matrix_U;
	p_amat_V = A_matrix_V;
	p_amat_D = A_matrix_D;
	for(i=0; i<bg_size; i++)
	{
		p_amat_V->count = p_amat->count;
		p_amat_U->count = p_amat->count;
		p_amat_D->count = p_amat->count;
		for(j=0; j<p_amat_U->count; j++)
		{
			p_amat_U->index[j] = p_amat->index[j];
			p_amat_U->value[j] = p_amat->value[j];
			p_amat_V->index[j] = p_amat->index[j];
			p_amat_V->value[j] = p_amat->value[j];
			p_amat_D->index[j] = p_amat->index[j];
			p_amat_D->value[j] = p_amat->value[j];
		}
		p_amat++;
		p_amat_U++;
		p_amat_V++;
		p_amat_D++;
	}
}

void BilateralGrid::construct_AMatrix_step2()
{
	int i, j;
	float *uv_pix;
	st_splat * p_splat;
	st_A * p_amat;
	st_A * p_amat_U;
	st_A * p_amat_V;

	p_amat = A_matrix;
	p_amat_U = A_matrix_U;
	p_amat_V = A_matrix_V;
	/*初期化*/
	for(i=0; i<bg_size; i++)
	{
		b_vecter_U[i] = 0;
		b_vecter_V[i] = 0;
		for(j=0; j<p_amat_U->count; j++)
		{
			p_amat_U->value[j] = p_amat->value[j];
			p_amat_V->value[j] = p_amat->value[j];
		}
		p_amat++;
		p_amat_U++;
		p_amat_V++;
	}

	uv_pix = mat_color.ptr<float>(0, 0) + 1;
	p_splat = splat_matrix;
	for(i=0; i<img_size; i++)
	{
		if(abs(*uv_pix - 0.5) > 0.001)
		{
			/*U成分の信頼度を加算*/
			A_matrix_U[p_splat->bg_index].value[0] += 1.0;
		}
		b_vecter_U[p_splat->bg_index] += (*uv_pix - 0.5);
		uv_pix++;

		if(abs(*uv_pix - 0.5) > 0.001)
		{
			/*V成分の信頼度を加算*/
			A_matrix_V[p_splat->bg_index].value[0] += 1.0;
		}
		b_vecter_V[p_splat->bg_index] += (*uv_pix - 0.5);
		uv_pix += 2;	/*yチャンネルは飛ばして、uチャンネルへ*/
		p_splat++;
	}

	/*multiply lambda*/
	p_amat_U = A_matrix_U;
	p_amat_V = A_matrix_V;
	element_num = 0;
	for(i=0; i<bg_size; i++)
	{
		for(j=0; j<p_amat_U->count; j++)
		{
			p_amat_U->value[j] *= LAMBDA;
			p_amat_V->value[j] *= LAMBDA;
			element_num++;
		}
		p_amat_U++;
		p_amat_V++;
	}
}


void BilateralGrid::construct_AMatrix_step2_for_depth()
{
	int i, j;
	float *d_pix;
	st_splat * p_splat;
	st_A * p_amat;
	st_A * p_amat_D;

	p_amat = A_matrix;
	p_amat_D = A_matrix_D;
	/*初期化*/
	for(i=0; i<bg_size; i++)
	{
		b_vecter_D[i] = 0;
		for(j=0; j<p_amat_D->count; j++)
		{
			p_amat_D->value[j] = p_amat->value[j];
		}
		p_amat++;
		p_amat_D++;
	}

	d_pix = mat_depth.ptr<float>(0, 0);
	p_splat = splat_matrix;
	for(i=0; i<img_size; i++)
	{
		if(abs(*d_pix - 0.5) > 0.001)
		{
			/*U成分の信頼度を加算*/
			A_matrix_D[p_splat->bg_index].value[0] += 1.0;
		}
		b_vecter_D[p_splat->bg_index] += (*d_pix - 0.5);
		d_pix++;
		p_splat++;
	}

	/*multiply lambda*/
	p_amat_D = A_matrix_D;
	element_num = 0;
	for(i=0; i<bg_size; i++)
	{
		for(j=0; j<p_amat_D->count; j++)
		{
			p_amat_D->value[j] *= LAMBDA;
			element_num++;
		}
		p_amat_D++;
	}
}





/****************************************************
brief	: ぼかし行列を作成する
note	:
*****************************************************/
void BilateralGrid::construct_BlurMatrix()
{
	int i, j, k;
	int count;
	int tmp_sum1;
	int tmp_sum2;

	st_table * p_table;
	st_table * p_table_ncol;
	st_table * p_table_nrow;

	//ポインタの設定
	p_table = table;
	p_table_ncol = table + 1;		//次の列 ncol = next column
	p_table_nrow = table + bg_step;	//次の行 nrow = next row

	blur_matrix = new st_blur [bg_size];
	for(i=0; i<bg_size; i++)
	{
		//自分自身をblur行列に加える
		blur_matrix[i].count = 1;
		blur_matrix[i].index[0] = i;
	}

	for(i=0; i<tbl_size - bg_step; i++)
	{
		//強度成分のぼかしを設定
		for(j=0; j<p_table->count-1; j++)
		{
			for(k=j+1; k<p_table->count; k++)
			{
				if(abs(p_table->data[j] - p_table->data[k]) < BLUR_RADIUS+1)
				{
					tmp_sum1 = p_table->sum + j;
					tmp_sum2 = p_table->sum + k;
					blur_matrix[tmp_sum1].index[blur_matrix[tmp_sum1].count] = tmp_sum2;
					blur_matrix[tmp_sum1].count++;
					blur_matrix[tmp_sum2].index[blur_matrix[tmp_sum2].count] = tmp_sum1;
					blur_matrix[tmp_sum2].count++;
				}
			}
		}
		//空間方向のぼかしを設定
		for(j=0; j<p_table->count; j++)
		{
			tmp_sum1 = p_table->sum + j;
			//列方向
			for(k=0; k<p_table_ncol->count; k++)
			{
				if(abs(p_table->data[j] - p_table_ncol->data[k]) == 0)
				{
					tmp_sum2 = p_table_ncol->sum + k;
					blur_matrix[tmp_sum1].index[blur_matrix[tmp_sum1].count] = tmp_sum2;
					blur_matrix[tmp_sum1].count++;
					blur_matrix[tmp_sum2].index[blur_matrix[tmp_sum2].count] = tmp_sum1;
					blur_matrix[tmp_sum2].count++;

				}
			}
			//行方向
			for(k=0; k<p_table_nrow->count; k++)
			{
				if(abs(p_table->data[j] - p_table_nrow->data[k]) == 0)
				{
					tmp_sum2 = p_table_nrow->sum + k;
					blur_matrix[tmp_sum1].index[blur_matrix[tmp_sum1].count] = tmp_sum2;
					blur_matrix[tmp_sum1].count++;
					blur_matrix[tmp_sum2].index[blur_matrix[tmp_sum2].count] = tmp_sum1;
					blur_matrix[tmp_sum2].count++;
				}
			}
		}
		p_table++;
		p_table_ncol++;
		p_table_nrow++;
	}
}


/****************************************************
brief	: デバック用。
note	: バイラテラルグリッドのぼかし効果を確認できる
*****************************************************/
void BilateralGrid::execute_Filter()
{
	int i,j;
	float *y_pix;
	st_calc * tmp_calc;
	st_calc * tmp_calc_blur;
	st_splat * p_splat;

	/*初期化*/
	tmp_calc = new st_calc [bg_size];
	tmp_calc_blur = new st_calc [bg_size];
	for(i=0; i<bg_size; i++)
	{
		tmp_calc[i].value = 0;
		tmp_calc[i].count = 0;
	}

	//Splat :sum the units in  gird
	y_pix = mat_input.ptr<float>(0, 0);
	p_splat = splat_matrix;
	for(i=0; i<img_size; i++)
	{
		tmp_calc[p_splat->bg_index].value += *y_pix;
		tmp_calc[p_splat->bg_index].count ++;
		y_pix++;
		p_splat++;
	}

	/*blur処理*/
	for(i=0; i<bg_size; i++)
	{
		//tmp_calcの値とst_blurの値を使ってぼかす
		tmp_calc_blur[i].value = WEIGHT_CENTER*tmp_calc[blur_matrix[i].index[0]].value;
		tmp_calc_blur[i].count = WEIGHT_CENTER*tmp_calc[blur_matrix[i].index[0]].count;
		for(j=1; j<blur_matrix[i].count; j++)
		{
			tmp_calc_blur[i].value += tmp_calc[blur_matrix[i].index[j]].value;
			tmp_calc_blur[i].count += tmp_calc[blur_matrix[i].index[j]].count;
		}
	}

	/*slice処理*/
	y_pix = mat_output.ptr<float>(0, 0);
	p_splat = splat_matrix;
	for(i=0; i<img_size; i++)
	{
		*y_pix = tmp_calc_blur[p_splat->bg_index].value / tmp_calc_blur[p_splat->bg_index].count;
		y_pix++;
		p_splat++;
	}
}

/****************************************************
brief	: 画像の確認。デバック用。
note	:
*****************************************************/
void BilateralGrid::show_Image(int num)
{
	//namedWindow("input", WINDOW_AUTOSIZE);
	switch(num)
	{
		case BG_INPUT:
			imshow("input", mat_input);
			break;
		case BG_OUTPUT:
			imshow("output", mat_color);
			break;
		case BG_DEPTH:
			imshow("depth", mat_depth);
			break;
		case BG_COLORIZED:
			imshow("output", mat_colorized);
			break;
		case BG_DEPTHSUPERRESOLUTED:
			imshow("depthsuperresolution", mat_depthsuperresoluted);
			break;
		default:
			break;
	}
	waitKey();
}


/****************************************************
brief	: 画像の出力
note	:
*****************************************************/
Mat3f BilateralGrid::get_Image(int num)
{
	Mat3f ret;
	switch(num)
	{
		case BG_COLORIZED:
			ret = mat_colorized;
			break;
		case BG_DEPTHSUPERRESOLUTED:
			ret = mat_depthsuperresoluted;
			break;
		default:
			break;
	}
	return ret;
}


/****************************************************
brief	: スライス行列の作成
note	:
*****************************************************/
void BilateralGrid::construct_SliceMatrix()
{
	int i, j, k;
	int bg_index;
	int bg_sum;
	int bg_i;
	int bg_j;
	int i_up;
	int j_up;
	float *y_pix;
	int sigma_2 = SIGMA*SIGMA;

	int index=0;
	st_index * tmp_splat;
	st_index * p_tmp;
	st_splat * p_splat;

	int comp;

	img_rows = (int)mat_input.rows;
	img_cols = (int)mat_input.cols;
	img_size = img_rows * img_cols;
	bg_step = (img_cols + SIGMA/2) / SIGMA;
	tbl_size = (bg_step) *(1 + (img_rows + SIGMA/2)/SIGMA) + 1;

	y_pix = mat_input.ptr<float>(0, 0);

	tmp_splat = new st_index [img_size];
	splat_matrix = new st_splat[img_size];
	table = new st_table [tbl_size];
	p_tmp = tmp_splat;
	p_splat = splat_matrix;
	/*初期化*/
	for(i=0; i<tbl_size; i++)
	{
		table[i].count = 0;
	}

	for(i=0, i_up=SIGMA/2; i<img_rows; i++, i_up++)
	{
		bg_i = i_up/SIGMA;

		for(j=0, j_up=SIGMA/2; j<img_cols; j++, j_up++)
		{
			bg_j = j_up/SIGMA;

			bg_index = bg_i*bg_step + bg_j;
			comp = (*y_pix * 255)/SIGMA;
			//y方向の大きさを考慮
			for(k=0; k<table[bg_index].count; k++)
			{
				if(table[bg_index].data[k] == comp)
				{
					break;
				}
			}
			if(k == table[bg_index].count)
			{
				table[bg_index].count++;
				table[bg_index].data[k] = comp;
			}
			p_tmp->row_index = k;
			p_tmp->col_index = bg_index;

			p_tmp++;
			index++;
			y_pix++;
		}
	}

	bg_sum = 0;
	table[0].sum = 0;
	for(i=0; i<tbl_size-1; i++)
	{
		table[i+1].sum = table[i].sum + table[i].count;
	}
	bg_size = table[tbl_size-1].sum + table[tbl_size-1].count; //gridの合計

	p_tmp = tmp_splat;
	p_splat = splat_matrix;
	for(i=0; i<img_rows; i++)
	{
		for(j=0; j<img_cols; j++)
		{
			p_splat->col = j;
			p_splat->row = i;
			p_splat->bright = table[p_tmp->col_index].data[p_tmp->row_index];
			p_splat->bg_index = table[p_tmp->col_index].sum + p_tmp->row_index;
			p_tmp++;
			p_splat++;
		}

	}
	delete tmp_splat;
}


/****************************************************
brief	: ユーザが着色した画像からYchを取得する
note	:
*****************************************************/
Mat1f BilateralGrid::get_Ych(Mat3f yuv)
{
	int y, x, c;
	Mat1f ret(yuv.rows, yuv.cols, CV_32FC1);
	float *yuv_pix;
	float *ret_pix;
	yuv_pix = yuv.ptr<float>(0, 0);
	ret_pix = ret.ptr<float>(0, 0);

	for(y=0; y<yuv.rows; y++)
	{
		for(x=0; x<yuv.cols; x++)
		{
			*ret_pix = *yuv_pix;
			yuv_pix += 3;
			ret_pix ++;
		}
	}
	return ret;
}

/****************************************************
brief	: 二重確率化処理
note	:
*****************************************************/
void BilateralGrid::calc_Bistochastic()
{
	int i, j, k;
	int * vect_m;
	float * vect_n;
	float * vect_numer;
	float * vect_denom;
	float sum;
	float tmp;

	st_splat * p_splat;

	vect_m = new int [bg_size]();
	vect_n = new float [bg_size]();
	vect_numer = new float [bg_size]();
	vect_denom = new float [bg_size]();
	p_splat = splat_matrix;

	//初期ベクトルの生成
	for(i=0; i<bg_size; i++)
	{
		vect_n[i] = 1;
		vect_m[i] = 0;
	}
	for(i=0; i<img_size; i++)
	{
		vect_m[p_splat->bg_index]++;
		p_splat++;
	}

	//収束するまでループする
	for(k = 0; k < BISTOCHASTIC_LOOP_MAX ; k++)
	{
		//分子と分母の計算
		for(i=0; i<bg_size; i++)
		{
			vect_numer[i] = vect_n[i] * vect_m[i];
		}
		for(i=0; i<bg_size; i++)
		{
			vect_denom[i] = WEIGHT_CENTER * vect_n[blur_matrix[i].index[0]];
			for(j=1; j<blur_matrix[i].count; j++)
			{
				vect_denom[i] += vect_n[blur_matrix[i].index[j]];
			}
		}

		sum = 0;
		//ベクトルnを更新する
		for(i=0; i<bg_size; i++)
		{
			tmp = sqrt(vect_numer[i] / vect_denom[i]);
			sum += abs(vect_n[i] - tmp);
			vect_n[i] = tmp;
		}

		//誤差がしきい値より小さいなら収束と判定
		if(sum < BISTOCHASTIC_THRE)
		{
			break;
		}
	}

	//収束した結果を格納する
	diagN_matrix = new float [bg_size]();
	diagM_matrix = new float [bg_size]();
	for(i=0; i<bg_size; i++)
	{
		diagN_matrix[i] = vect_n[i];
		diagM_matrix[i] = vect_m[i];
	}
}


/****************************************************
brief	: フォーマットをCSR形式にする。
note	: ここまでは、オリジナル形式で作成していたので。
*****************************************************/
str_CSR	BilateralGrid::convertCSR(st_A * mat_A)
{
	int i, j;
	int k = 0;
	int	flg_insart;
	st_A * p_amat;
	p_amat = mat_A;
	str_CSR csr;

	csr.val = new double[element_num];
	csr.col_index = new int [element_num];
	csr.row_index = new int [bg_size+1];	//　データの数
	csr.str_size = element_num;	//データの数
	csr.row_size = bg_size;
	csr.col_size = bg_size;

	csr.row_index[0] = 1;

	for(i=0; i<bg_size; i++)
	{
		flg_insart = 0;
		for(j=1; j<p_amat->count; j++)
		{
			if(p_amat->index[j] >= i && flg_insart == 0)
			{
				flg_insart = 1;
				csr.val[k] = p_amat->value[0];
				csr.col_index[k] = p_amat->index[0];
				k++;
			}
			csr.val[k] = p_amat->value[j];
			csr.col_index[k] = p_amat->index[j];
			k++;
		}
		if(flg_insart == 0)
		{
			csr.val[k] = p_amat->value[0];
			csr.col_index[k] = p_amat->index[0];
			k++;
		}
		p_amat++;
		csr.row_index[i+1] = k+1;
	}

	return csr;
}


/****************************************************
brief	: 共役勾配法を実行、スライス処理までやってしまう
note	:
*****************************************************/
void BilateralGrid::execute_ICCG(int iter, float eps)
{
	#ifdef _DEBUG_
	std::cout << "start execute_ICCG" << std::endl;
	#endif
	float *uv_pix;
	st_splat * p_splat;
	vector<double> vec_u_in(bg_size);
	vector<double> vec_v_in(bg_size);
	vector<double> vec_u_out(bg_size);
	vector<double> vec_v_out(bg_size);
	int loop_cut;

	#ifdef _DEBUG_
	std::cout << "for bg_size" << std::endl;
	#endif
	for(int i=0; i<bg_size; i++)
	{
		vec_u_in[i] = b_vecter_U[i];
		vec_v_in[i] = b_vecter_V[i];
	}

	loop_cut = 14 * (img_cols/SIGMA);	/*ぼかし行列のサイズ 14 = 7*2 */

	#ifdef _DEBUG_
	std::cout << "convertCSR 1" << std::endl;
	#endif
	mat_A_csr = convertCSR(A_matrix_U);
	#ifdef _DEBUG_
	std::cout << "pre_ICD" << std::endl;
	#endif
	csr_col = pre_ICD(&mat_A_csr);

	#ifdef _DEBUG_
	std::cout << "ICCGSolver 1" << std::endl;
	#endif
	ICCGSolver( &mat_A_csr , vec_u_in, vec_u_out, iter, eps, csr_col);
	#ifdef _DEBUG_
	std::cout << "convertCSR 2" << std::endl;
	#endif
	mat_A_csr = convertCSR(A_matrix_V);
	#ifdef _DEBUG_
	std::cout << "ICCGSolver 2" << std::endl;
	#endif
	ICCGSolver( &mat_A_csr , vec_v_in, vec_v_out, iter, eps, csr_col);

	uv_pix = mat_colorized.ptr<float>(0, 0)+1;
	p_splat = splat_matrix;
	#ifdef _DEBUG_
	std::cout << "for img_size" << std::endl;
	#endif
	for(int i=0; i<img_size; i++)
	{
		*uv_pix = vec_u_out[p_splat->bg_index]+ 0.5;
		uv_pix++;
		*uv_pix = vec_v_out[p_splat->bg_index]+ 0.5;
		uv_pix += 2;
		p_splat++;
	}
	#ifdef _DEBUG_
	std::cout << "cvtColor" << std::endl;
	#endif
	cvtColor(mat_colorized, mat_colorized, CV_YCrCb2BGR);
}


void BilateralGrid::execute_ICCG_for_depth(int iter, float eps)
{
	#ifdef _DEBUG_
	std::cout << "start execute_ICCG" << std::endl;
	#endif
	float *d_pix;
	st_splat * p_splat;
	vector<double> vec_d_in(bg_size);
	vector<double> vec_d_out(bg_size);
	int loop_cut;

	#ifdef _DEBUG_
	std::cout << "for bg_size" << std::endl;
	#endif
	for(int i=0; i<bg_size; i++)
	{
		vec_d_in[i] = b_vecter_D[i];
	}

	loop_cut = 14 * (img_cols/SIGMA);	/*ぼかし行列のサイズ 14 = 7*2 */

	#ifdef _DEBUG_
	std::cout << "convertCSR 1" << std::endl;
	#endif
	mat_A_csr = convertCSR(A_matrix_D);
	#ifdef _DEBUG_
	std::cout << "pre_ICD" << std::endl;
	#endif
	csr_col = pre_ICD(&mat_A_csr);

	#ifdef _DEBUG_
	std::cout << "convertCSR" << std::endl;
	#endif
	mat_A_csr = convertCSR(A_matrix_D);
	#ifdef _DEBUG_
	std::cout << "ICCGSolver 2" << std::endl;
	#endif
	ICCGSolver( &mat_A_csr , vec_d_in, vec_d_out, iter, eps, csr_col);

	d_pix = mat_depthsuperresoluted.ptr<float>(0, 0);
	p_splat = splat_matrix;
	#ifdef _DEBUG_
	std::cout << "for img_size" << std::endl;
	#endif
	for(int i=0; i<img_size; i++)
	{
		// std::cout << "d_pix"<<vec_d_out[p_splat->bg_index] << std::endl;
		*d_pix = vec_d_out[p_splat->bg_index]+ 0.5;
		d_pix++;
		p_splat++;
	}
	#ifdef _DEBUG_
	std::cout << "cvtColor" << std::endl;
	#endif

	// cvtColor(mat_depthsuperresoluted, mat_depthsuperresoluted, CV_YCrCb2BGR);
}
