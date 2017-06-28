#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "InputImage.hpp"

static int glb_mouse_x;
static int glb_mouse_y;
static bool glb_mouse_click = false;
static bool glb_mouse_left = false;

/****************************************************
brief	:	RGB画像の3ch全てにGlay画像の1chを代入する
note	:	opencvの仕様上?、しぶしぶクラスから外すことに。
*****************************************************/
static void my_mouse_callback(int event, int x, int y, int flags, void* param)
{
	switch (event){
		case EVENT_MOUSEMOVE:
			if (glb_mouse_click){
				glb_mouse_x = x;
				glb_mouse_y = y;
			}
			break;

		case EVENT_LBUTTONDOWN:
			glb_mouse_click = true;
			glb_mouse_x = x;
			glb_mouse_y = y;
			break;

		case EVENT_LBUTTONUP:
			glb_mouse_left = true;
			glb_mouse_click = false;
			break;
	}
}

/****************************************************
brief	: コンストラクタ
note	:
*****************************************************/
// InputImage::InputImage(Mat1f mat_image)
// {
// 	mat_input = mat_image.clone();
// 	cvtColor(mat_input, mat_gray, COLOR_BGR2GRAY);
// 	mat_draw_bp = copy_GlaychForRGBch(mat_gray, mat_input);
// 	cvtColor(mat_draw_bp, mat_yuv, COLOR_BGR2YCrCb);
// 	mat_draw = mat_draw_bp.clone();
// }


InputImage::InputImage(Mat3f mat_image)
{
	mat_input = mat_image.clone();
	cvtColor(mat_input, mat_gray, COLOR_BGR2GRAY);
	mat_draw_bp = copy_GlaychForRGBch(mat_gray, mat_input);
	// cvtColor(mat_draw_bp, mat_yuv, COLOR_BGR2YCrCb);
	cvtColor(mat_image, mat_yuv, COLOR_BGR2YCrCb);
	std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
	std::cout << mat_draw_bp.cols<< "x" << mat_draw_bp.rows <<"x"<< mat_draw_bp.channels()<< std::endl;
	std::cout << mat_image.cols<< "x" << mat_image.rows << "x"<< mat_image.channels()<<std::endl;
	std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
	mat_draw = mat_draw_bp.clone();
}


/****************************************************
brief	:	RGB画像の3ch全てにGlay画像の1chを代入する
note	:
*****************************************************/
Mat3f InputImage::copy_GlaychForRGBch(Mat1f gray, Mat3f color)
{
	int y, x, c;
	float* gray_pix;
	float* color_pix;
	Mat3f ret = color.clone();
	gray_pix = gray.ptr<float>(0, 0);
	color_pix = ret.ptr<float>(0, 0);

	for(y=0; y<gray.rows; y++)
	{
		for(x=0; x<gray.cols; x++)
		{
			for(c=0; c<color.channels(); c++)
			{
				*color_pix = *gray_pix;
				color_pix++;
			}
			gray_pix++;
		}
	}
	return ret;
}

/****************************************************
brief	:	Gray画像に色をつける
note	:
*****************************************************/
void InputImage::draw_Trajectory(Mat3f* img)
{
	int i, j;
	float red, green, blue;
	int y, x;
	int r = MARK_RADIUS;
	int r2 = r * r;
	float* color_pix;

	y = mouse_y - r;
	for(i=-r; i<r+1 ; i++, y++)
	{
		x = mouse_x - r;
		color_pix = mat_input.ptr<float>(y, x);
		for(j=-r; j<r+1; j++, x++)
		{
			//マークを円形にする
			if(i*i + j*j > r2)
			{
				color_pix += mat_input.channels();
				continue;
			}

			//境界条件を意識
			if(y<0 || y>=mat_input.rows || x<0 || x>=mat_input.cols)
			{
				break;
			}

			blue = *color_pix;
			color_pix++;
			green = *color_pix;
			color_pix++;
			red = *color_pix;
			color_pix++;
			circle(*img, Point2d(x, y), 0, Scalar(blue, green, red), -1);
		}
	}
}


/****************************************************
brief	:	デバック用、画像表示
note	:
*****************************************************/
void InputImage::show_Image(int num)
{
	namedWindow("input", WINDOW_AUTOSIZE);
	switch(num)
	{
		case IMG_INPUT:
			imshow("input", mat_input);
			break;
		case IMG_GRAY:
			imshow("input", mat_gray);
			break;
		case IMG_DRAW:
			imshow("input", mat_draw);
			break;
		case IMG_YUV:
			imshow("input", mat_yuv);
			break;
		case IMG_DRAWYUV:
			imshow("input", mat_draw_yuv);
			break;
		default:
			break;
	}
	waitKey();
}


/****************************************************
brief	:	画像データのゲッター
note	:
*****************************************************/
Mat3f InputImage::get_Image(int num)
{
	switch(num)
	{
		case IMG_DRAW:
			return mat_draw;
			break;
		case IMG_YUV:
			return mat_yuv;
			break;
		case IMG_DRAWYUV:
			return mat_draw_yuv;
			break;
		default:
			return mat_input;
			break;
	}
}


/****************************************************
brief	: 画像に色を塗る
note	: もともと色の付いている画像の色を戻すだけ
*****************************************************/
void InputImage::draw_Image(void)
{
	//mat_draw = mat_draw_bp.clone();
	namedWindow("draw", WINDOW_AUTOSIZE);
	imshow("draw", mat_draw);
	setMouseCallback("draw", my_mouse_callback, (void *)&mat_draw);
	while (1){
		mouse_x = glb_mouse_x;
		mouse_y = glb_mouse_y;
		mouse_click = glb_mouse_click;
		mouse_left = glb_mouse_left;

		// マウスのクリックを押している間、軌跡を模写する
		if (mouse_click) {
			draw_Trajectory(&mat_draw);
			imshow("draw", mat_draw);
		}
		// Escで終了
		if (waitKey(2) == 27)
			break;
	}
	cvtColor(mat_draw, mat_draw_yuv, COLOR_BGR2YCrCb);
	waitKey();
}
