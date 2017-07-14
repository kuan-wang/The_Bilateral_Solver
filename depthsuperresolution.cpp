#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <time.h>
#include "InputImage.hpp"
#include "BilateralGrid.hpp"

#define ICCG_LOOP_MAX	(200)
#define ICCG_EPS		(0.01)

using namespace cv;
using namespace std;

const String keys =
	"{help h usage |      | print this message   }"
	"{@image      |      | image for input   }";

int main(int argc, char **argv)
{
    clock_t now;
    now = clock();
    printf( "start depth_superres : now is %f seconds\n\n", (double)(now) / CLOCKS_PER_SEC);

    String imgName_T;
    String imgName_R;
    String imgName_C;
	Mat3f mat_R;
	Mat3f mat_T;
	Mat3f mat_bg_in;

    // CommandLineParser parser(argc, argv, keys);
	// if(parser.has("h") || argc != 2){
	// 	parser.printMessage();
	// 	return 0;
	// }
	//
    // imgName_R = parser.get<String>(0);
    imgName_R = "reference.png";
	imgName_T = "target.png";
    // imgName_T = parser.get<String>(1);
    // imgName_C = parser.get<String>(2);
	std::cout << "imgName:" << imgName_T << imgName_R<< std::endl;

	mat_R= imread(imgName_R, 1)/255;
	mat_T= imread(imgName_T, 0)/255;
	// std::cout << "mat_in:"<<mat_in.cols<<"x"<<mat_in.rows << std::endl;

	InputImage InImg(mat_R);
	mat_bg_in = InImg.get_Image(IMG_YUV);
	// cvtColor(mat_bg_in, mat_bg_in, COLOR_YCrCb2BGR);

	BilateralGrid BiGr(mat_bg_in);

	imshow("mat_R",mat_R);
	BiGr.Depthsuperresolution(mat_R,mat_T,5,5,5);

    now = clock();
    printf( "end depthsuperresolution : now is %f seconds\n\n", (double)(now) / CLOCKS_PER_SEC);

	// InputImage InImg(mat_in);
	// mat_bg_in = InImg.get_Image(IMG_YUV);
	// // cvtColor(mat_bg_in, mat_bg_in, COLOR_YCrCb2BGR);
	//
	// BilateralGrid BiGr(mat_bg_in);
	// // BiGr.construct_SliceMatrix();
	// BiGr.construct_SliceMatrix_for_depth();
	// BiGr.construct_BlurMatrix();
	// BiGr.calc_Bistochastic();
	// BiGr.construct_AMatrix_step1();
	//
	// cout << "process" << endl;
	// BiGr.set_DepthImage(mat_bg_draw_in);
	// cout << "construct_AMatrix_step2" << endl;
	//
	//
	// BiGr.construct_AMatrix_step2_for_depth();
	// cout << "execute_ICCG" << endl;
	//
	//
	// BiGr.execute_ICCG_for_depth(ICCG_LOOP_MAX, ICCG_EPS);
	// cout << "show_Image" << endl;
	// BiGr.show_Image(BG_DEPTHSUPERRESOLUTED);

}
