#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "ICCG.hpp"

#define SIGMA (5)
#define BLUR_RADIUS (1)
#define WEIGHT_CENTER (6)
#define WEIGHT_NEIGHBOR (1)
#define BISTOCHASTIC_LOOP_MAX (50)
#define BISTOCHASTIC_THRE (0.001)
#define LAMBDA (1.0)

using namespace cv;
using namespace std;

enum ShowBgImg
{
	BG_INPUT = 0,
	BG_OUTPUT,
	BG_DEPTH,
	BG_COLORIZED,
	BG_DEPTHSUPERRESOLUTED,
	BG_NUM
};

struct st_index
{
	int row_index;
	int col_index;
};

struct st_table
{
	int count;
	int sum;
	int data[SIGMA*SIGMA];
};

struct st_splat
{
	int col;
	int row;
	int bright;
	int bg_index;
};

struct st_calc
{
	float value;
	int count;
};

struct st_blur
{
	int count;
	int index[BLUR_RADIUS*2*3+1];
};

struct st_A
{
	int count;
	int index[BLUR_RADIUS*2*3+1];
	float value[BLUR_RADIUS*2*3+1];
};


/****************************************************
brief	:	Bilateral Gridクラス
note	:
*****************************************************/
class BilateralGrid
{
	public:
		BilateralGrid(Mat3f mat_image);
		void set_DrawImage(Mat3f mat_draw_image);
		void set_DepthImage(Mat1f mat_depth_image);
		void construct_SliceMatrix(void);
		void construct_BlurMatrix(void);
		void construct_AMatrix_step1(void);
		void construct_AMatrix_step2(void);
		void construct_AMatrix_step2_for_depth();

		void show_Image(int num);
		void execute_Filter(void);
		void calc_Bistochastic(void);
		void execute_Bistochastic(void);
		void execute_ICCG(int iter, float eps);
		void execute_ICCG_for_depth(int iter, float eps);
		Mat3f get_Image(int num);

	private:
		str_CSR	convertCSR(st_A*);
		str_CSR convertCSR();
		Mat1f get_Ych(Mat3f);
		Mat1f mat_input;
		Mat1f mat_output;
		Mat1f mat_depth;
		Mat1f mat_depthsuperresoluted;
		Mat3f mat_color;
		Mat3f mat_colorized;
		float **grid;		//bilateral grid
		int img_cols;
		int img_rows;
		int img_size;
		int tbl_size;
		int bg_size;
		int bg_step;
		int element_num;
		str_CSR mat_A_csr;
		str_CSR_colsort * csr_col;

		float * diagN_matrix;
		float * diagM_matrix;
		float * b_vecter_U;
		float * b_vecter_V;
		float * b_vecter_D;
		float * b_vecter_U_count;
		float * b_vecter_V_count;
		float * b_vecter_D_count;
		st_splat * splat_matrix;	//img_size vector
		st_table * table;
		st_blur * blur_matrix;
		st_A * A_matrix;
		st_A * A_matrix_U;
		st_A * A_matrix_V;
		st_A * A_matrix_D;
};
