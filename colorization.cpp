#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
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

    String imgName_T;
    String imgName_R;
    String imgName_C;
	Mat3f mat_in;
	Mat3f mat_in_r;
	Mat3f mat_bg_in;
	Mat3f mat_bg_draw_in;
	Mat3f mat_bg_depth_in;
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

	mat_in= imread(argv[1], 1)/255;
	mat_in_r= imread(argv[2], 1)/255;

	// InputImage InImg(mat_in);
	InputImage InImg(mat_in, mat_in_r);
	mat_bg_in = InImg.get_Image(IMG_YUV);
	InImg.draw_Image();
	mat_bg_draw_in = InImg.get_Image(IMG_DRAWYUV);

	BilateralGrid BiGr(mat_bg_in);

	BiGr.Colorization(mat_in,mat_bg_draw_in);

	imwrite("draw.png" , InImg.get_Image(IMG_DRAW)*255);




}
