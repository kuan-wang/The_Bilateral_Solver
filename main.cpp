#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "InputImage.hpp"
#include "BilateralGrid.hpp"


using namespace cv;
using namespace std;

const String keys =
	"{help h usage |      | print this message   }"
	"{@image      |      | image for input   }";

int main(int argc, char **argv)
{
    String imgName_T;
    String imgName_R;
    String imgName_C;
	Mat3f mat_in;
	Mat3f mat_bg_in;
	Mat3f mat_bg_draw_in;
	Mat1f mat_bg_depth_in;

    CommandLineParser parser(argc, argv, keys);
	if(parser.has("h") || argc != 2){
		parser.printMessage();
		return 0;
	}

    imgName_R = parser.get<String>(0);
    imgName_R = "reference.png";
	imgName_T = "target.png";
    // imgName_T = parser.get<String>(1);
    // imgName_C = parser.get<String>(2);
	std::cout << "imgName:" << imgName_T << imgName_R<< std::endl;

	mat_in= imread(imgName_R, 1)/255;
	// std::cout << "mat_in:"<<mat_in.cols<<"x"<<mat_in.rows << std::endl;

	imshow("mat_in",mat_in);
	//入力画像作成用のクラス
	InputImage InImg(mat_in);
	mat_bg_in = InImg.get_Image(IMG_YUV);
	// cvtColor(mat_bg_in, mat_bg_in, COLOR_YCrCb2BGR);
	imshow("mat_bg_in",mat_bg_in);

	//初期セットアップ
	BilateralGrid BiGr(mat_bg_in);
	// BiGr.construct_SliceMatrix();
	BiGr.construct_SliceMatrix_for_depth();
	BiGr.construct_BlurMatrix();
	BiGr.calc_Bistochastic();
	BiGr.construct_AMatrix_step1();
	cout << "Bistochastic Fin" << endl;

	// InImg.draw_Image();
	// mat_bg_draw_in = InImg.get_Image(IMG_DRAWYUV);
	mat_bg_draw_in= imread(imgName_T, 0)/255;
	// cout << "Fin" << endl;
	// std::cout << "mat_in:"<<mat_bg_draw_in.col(0) << std::endl;
	// InputImage InImgT(mat_bg_draw_in);
	// mat_bg_draw_in = InImgT.get_Image(IMG_YUV);
	// imshow("mat_bg_draw_in",mat_bg_draw_in);
	cout << "process" << endl;
	BiGr.set_DepthImage(mat_bg_draw_in);
	cout << "construct_AMatrix_step2" << endl;


	BiGr.construct_AMatrix_step2_for_depth();
	cout << "execute_ICCG" << endl;

	// BiGr.show_Image(BG_DEPTHSUPERRESOLUTED);

	BiGr.execute_ICCG_for_depth(ICCG_LOOP_MAX, ICCG_EPS);
	cout << "show_Image" << endl;
	BiGr.show_Image(BG_DEPTHSUPERRESOLUTED);

	// imwrite("draw.png" , InImg.get_Image(IMG_DRAW)*255);
	// imwrite("depthsuperresoluted.jpg" , BiGr.get_Image(BG_DEPTHSUPERRESOLUTED)*255);




    // String imgName_T;
    // String imgName_R;
    // String imgName_C;
	// Mat3f mat_in;
	// Mat3f mat_bg_in;
	// Mat3f mat_bg_draw_in;
	// Mat3f mat_bg_depth_in;
	//
    // CommandLineParser parser(argc, argv, keys);
	// if(parser.has("h") || argc != 2){
	// 	parser.printMessage();
	// 	return 0;
	// }
	//
    // imgName_R = parser.get<String>(0);
	// imgName_T = "target.png";
    // // imgName_T = parser.get<String>(1);
    // // imgName_C = parser.get<String>(2);
	// std::cout << "imgName:" << imgName_T << imgName_R<< std::endl;
	//
	// mat_in= imread(imgName_R, 1)/255;
	//
	// //入力画像作成用のクラス
	// InputImage InImg(mat_in);
	// mat_bg_in = InImg.get_Image(IMG_YUV);
	//
	// //初期セットアップ
	// BilateralGrid BiGr(mat_bg_in);
	// BiGr.construct_SliceMatrix();
	// BiGr.construct_BlurMatrix();
	// BiGr.calc_Bistochastic();
	// BiGr.construct_AMatrix_step1();
	// cout << "Bistochastic Fin" << endl;
	//
	//
	//
	// InImg.draw_Image();
	// mat_bg_draw_in = InImg.get_Image(IMG_DRAWYUV);
	// cout << "process" << endl;
	// BiGr.set_DrawImage(mat_bg_draw_in);
	// cout << "construct_AMatrix_step2" << endl;
	// BiGr.construct_AMatrix_step2();
	// cout << "execute_ICCG" << endl;
	// BiGr.execute_ICCG(ICCG_LOOP_MAX, ICCG_EPS);
	// cout << "show_Image" << endl;
	// BiGr.show_Image(BG_COLORIZED);
	//
	// imwrite("draw.png" , InImg.get_Image(IMG_DRAW)*255);
	// imwrite("colorized.png" , BiGr.get_Image(BG_COLORIZED)*255);



}
