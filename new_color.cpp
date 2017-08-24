//
//
// #include<opencv2/core/core.hpp>
// #include<opencv2/highgui.hpp>
// #include<opencv2/opencv.hpp>
//
// #include <time.h>
// #include <chrono>
// #include <vector>
// #include <iostream>
// #include <opencv2/ximgproc.hpp>
//
// #define MARK_RADIUS 15
//
// static int glb_mouse_x;
// static int glb_mouse_y;
// static bool glb_mouse_click = false;
// static bool glb_mouse_left = false;
//
// static bool mouse_click;
// static bool mouse_left;
// static int mouse_x;
// static int mouse_y;
// cv::Mat mat_input_reference;
// cv::Mat mat_input_confidence;
//
//
//
// static void mouseCallback(int event, int x, int y, int flags, void* param)
// {
// 	switch (event){
// 		case cv::EVENT_MOUSEMOVE:
// 			if (glb_mouse_click){
// 				glb_mouse_x = x;
// 				glb_mouse_y = y;
// 			}
// 			break;
//
// 		case cv::EVENT_LBUTTONDOWN:
// 			glb_mouse_click = true;
// 			glb_mouse_x = x;
// 			glb_mouse_y = y;
// 			break;
//
// 		case cv::EVENT_LBUTTONUP:
// 			glb_mouse_left = true;
// 			glb_mouse_click = false;
// 			break;
// 	}
// }
//
// void draw_Trajectory_Byreference(cv::Mat* img)
// {
// 	int i, j;
// 	uchar red, green, blue;
// 	int y, x;
// 	int r = MARK_RADIUS;
// 	int r2 = r * r;
// 	uchar* color_pix;
//
// 	y = mouse_y - r;
// 	for(i=-r; i<r+1 ; i++, y++)
// 	{
// 		x = mouse_x - r;
// 		color_pix = mat_input_reference.ptr<uchar>(y, x);
// 		for(j=-r; j<r+1; j++, x++)
// 		{
// 			if(i*i + j*j > r2)
// 			{
// 				color_pix += mat_input_reference.channels();
// 				continue;
// 			}
//
// 			if(y<0 || y>=mat_input_reference.rows || x<0 || x>=mat_input_reference.cols)
// 			{
// 				break;
// 			}
//
// 			blue = *color_pix;
// 			color_pix++;
// 			green = *color_pix;
// 			color_pix++;
// 			red = *color_pix;
// 			color_pix++;
// 			cv::circle(*img, cv::Point2d(x, y), 0.1, cv::Scalar(blue, green, red), -1);
//       // mat_input_confidence.at<uchar>(x,y) = (blue + green + red)/3;
//       mat_input_confidence.at<uchar>(y,x) = 255;
// 		}
// 	}
// }
//
// cv::Mat copy_GlaychForRGBch(cv::Mat gray, cv::Mat color)
// {
// 	int y, x, c;
// 	uchar* gray_pix;
// 	uchar* color_pix;
// 	cv::Mat ret = color.clone();
// 	gray_pix = gray.ptr<uchar>(0, 0);
// 	color_pix = ret.ptr<uchar>(0, 0);
//
// 	for(y=0; y<gray.rows; y++)
// 	{
// 		for(x=0; x<gray.cols; x++)
// 		{
// 			for(c=0; c<color.channels(); c++)
// 			{
// 				*color_pix = *gray_pix;
// 				color_pix++;
// 			}
// 			gray_pix++;
// 		}
// 	}
// 	return ret;
// }
//
//
//
// int main(int argc, char const *argv[]) {
//
//     std::cout << "hello solver" << '\n';
//
//     float filtering_time;
//
//     clock_t now;
//     now = clock();
//     printf( "start : now is %f seconds\n\n", (float)(now) / CLOCKS_PER_SEC);
//
//     cv::Mat reference = cv::imread(argv[1],1);
//     cv::Mat input = cv::imread(argv[1],0);
//     // cv::Mat target = cv::imread(argv[2],1);
//     cv::Mat target;
//
//     float spatialSigma = float(atof(argv[2]));
//     float lumaSigma = float(atof(argv[3]));
//     float chromaSigma = float(atof(argv[4]));
//     std::cout << "reference:" << reference.cols<<"x"<< reference.rows<< std::endl;
//
//
//     cv::Mat mat_gray;
// 	  cv::cvtColor(reference, mat_gray, cv::COLOR_BGR2GRAY);
//     target = copy_GlaychForRGBch(mat_gray, reference);
//
//     cv::namedWindow("draw", cv::WINDOW_AUTOSIZE);
//     cv::imshow("draw", target);
//     cv::setMouseCallback("draw", mouseCallback, (void *)&target);
//     mat_input_reference = reference.clone();
//     mat_input_confidence = 0*cv::Mat::ones(mat_gray.size(),mat_gray.type());
//     // mat_input_confidence = mat_gray;
//     while (1)
//     {
//     		mouse_x = glb_mouse_x;
//     		mouse_y = glb_mouse_y;
//     		mouse_click = glb_mouse_click;
//     		mouse_left = glb_mouse_left;
//
//     		if (mouse_click)
//         {
//     			draw_Trajectory_Byreference(&target);
//     			cv::imshow("draw", target);
//     		}
//     		if (cv::waitKey(2) == 27)
//     		    break;
//     }
//     cv::cvtColor(target, target, cv::COLOR_BGR2YCrCb);
//
//     std::vector<cv::Mat> src_channels;
//     std::vector<cv::Mat> dst_channels;
//
//     cv::split(target,src_channels);
//
//     cv::Mat result1 = cv::Mat(input.size(),input.type());
//     cv::Mat result2 = cv::Mat(input.size(),input.type());
//   	std::chrono::steady_clock::time_point start_solver = std::chrono::steady_clock::now();
// //////////////////////////////////////////////////////////////////////////////////
// //////////////////////////////////////////////////////////////////////////////////
//     // cv::Mat x;
//     // cv::Mat w;
//     // cv::Mat xw;
//     // cv::Mat filtered_xw;
//     // cv::Mat filtered_w;
//     // cv::Mat filtered_disp;
//     //
//     // // cv::ximgproc::fastGlobalSmootherFilter(input, tu, result1, spatialSigma, lumaSigma);
//     // // cv::ximgproc::fastGlobalSmootherFilter(input, tv, result2, spatialSigma, lumaSigma);
//     // tu.convertTo(x, CV_32FC1, 1.0f/255.0f);
//     // cu.convertTo(w, CV_32FC1);
//     // xw = x.mul(w);
//     // cv::ximgproc::fastGlobalSmootherFilter(input, xw, filtered_xw, spatialSigma, lumaSigma);
//     // cv::ximgproc::fastGlobalSmootherFilter(input, w, filtered_w, spatialSigma, lumaSigma);
//     // cv::divide(filtered_xw, filtered_w, result1, 255.0f, CV_8UC1);
//     //
//     // tv.convertTo(x, CV_32FC1, 1.0f/255.0f);
//     // cv.convertTo(w, CV_32FC1);
//     // xw = x.mul(w);
//     // cv::ximgproc::fastGlobalSmootherFilter(input, xw, filtered_xw, spatialSigma, lumaSigma);
//     // cv::ximgproc::fastGlobalSmootherFilter(input, w, filtered_w, spatialSigma, lumaSigma);
//     // cv::divide(filtered_xw, filtered_w, result2, 255.0f, CV_8UC1);
// //////////////////////////////////////////////////////////////////////////////////
// //////////////////////////////////////////////////////////////////////////////////
//
//     dst_channels.push_back(src_channels[0]);
//     cv::ximgproc::fastBilateralSolverFilter(input,src_channels[1],mat_input_confidence,result1,spatialSigma,lumaSigma,chromaSigma);
//     dst_channels.push_back(result1);
//     cv::ximgproc::fastBilateralSolverFilter(input,src_channels[2],mat_input_confidence,result2,spatialSigma,lumaSigma,chromaSigma);
//     dst_channels.push_back(result2);
//
//     cv::merge(dst_channels,target);
//     cv::cvtColor(target, target, cv::COLOR_YCrCb2BGR);
//
//   	std::chrono::steady_clock::time_point end_solver = std::chrono::steady_clock::now();
//     std::cout << "solver time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_solver - start_solver).count() << "ms" << std::endl;
//
//
//
//     // cv::equalizeHist(result, result);
// 	  cv::imshow("input",input);
//   	cv::imshow("output",target);
//
// 	  cv::waitKey(0);
//
//
//     return 0;
// }



#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/utility.hpp"

#include <time.h>
#include <vector>
#include <iostream>
#include <opencv2/ximgproc.hpp>

using namespace cv;

#define MARK_RADIUS 15

static int glb_mouse_x;
static int glb_mouse_y;
static bool glb_mouse_click = false;
static bool glb_mouse_left = false;

static bool mouse_click;
static bool mouse_left;
static int mouse_x;
static int mouse_y;
cv::Mat mat_input_reference;
cv::Mat mat_input_confidence;


static void mouseCallback(int event, int x, int y, int flags, void* param)
{
	switch (event){
		case cv::EVENT_MOUSEMOVE:
			if (glb_mouse_click){
				glb_mouse_x = x;
				glb_mouse_y = y;
			}
			break;

		case cv::EVENT_LBUTTONDOWN:
			glb_mouse_click = true;
			glb_mouse_x = x;
			glb_mouse_y = y;
			break;

		case cv::EVENT_LBUTTONUP:
			glb_mouse_left = true;
			glb_mouse_click = false;
			break;
	}
}

void draw_Trajectory_Byreference(cv::Mat* img)
{
	int i, j;
	uchar red, green, blue;
	int y, x;
	int r = MARK_RADIUS;
	int r2 = r * r;
	uchar* color_pix;

	y = mouse_y - r;
	for(i=-r; i<r+1 ; i++, y++)
	{
		x = mouse_x - r;
		color_pix = mat_input_reference.ptr<uchar>(y, x);
		for(j=-r; j<r+1; j++, x++)
		{
			if(i*i + j*j > r2)
			{
				color_pix += mat_input_reference.channels();
				continue;
			}

			if(y<0 || y>=mat_input_reference.rows || x<0 || x>=mat_input_reference.cols)
			{
				break;
			}

			blue = *color_pix;
			color_pix++;
			green = *color_pix;
			color_pix++;
			red = *color_pix;
			color_pix++;
			cv::circle(*img, cv::Point2d(x, y), 0.1, cv::Scalar(blue, green, red), -1);
      // mat_input_confidence.at<uchar>(x,y) = (blue + green + red)/3;
      mat_input_confidence.at<uchar>(y,x) = 255;
		}
	}
}

cv::Mat copy_GlaychForRGBch(cv::Mat gray, cv::Mat color)
{
	int y, x, c;
	uchar* gray_pix;
	uchar* color_pix;
	cv::Mat ret = color.clone();
	gray_pix = gray.ptr<uchar>(0, 0);
	color_pix = ret.ptr<uchar>(0, 0);

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




const String keys =
    "{help h usage ?     |                | print this message                                                }"
    "{sigma_spatial      |8               | parameter of post-filtering                                       }"
    "{sigma_luma         |8               | parameter of post-filtering                                       }"
    "{sigma_chroma       |8               | parameter of post-filtering                                       }"
    ;



int main(int argc, char** argv)
{
    // CommandLineParser parser(argc,argv,keys);
    // parser.about("Disparity Filtering Demo");
    // if (parser.has("help"))
    // {
    //     parser.printMessage();
    //     return 0;
    // }
    //
    // String img = parser.get<String>(0);
    // double sigma_spatial  = parser.get<double>("sigma_spatial");
    // double sigma_luma  = parser.get<double>("sigma_luma");
    // double sigma_chroma  = parser.get<double>("sigma_chroma");


    float filtering_time;

    cv::Mat reference = cv::imread(argv[1],1);
    cv::Mat input = cv::imread(argv[1],0);
    // cv::Mat target = cv::imread(argv[2],1);
    cv::Mat target;

    float sigma_spatial = float(atof(argv[2]));
    float sigma_luma = float(atof(argv[3]));
    float sigma_chroma = float(atof(argv[4]));

    std::cout << "reference:" << reference.cols<<"x"<< reference.rows<< std::endl;


    cv::Mat mat_gray;
	  cv::cvtColor(reference, mat_gray, cv::COLOR_BGR2GRAY);
    target = copy_GlaychForRGBch(mat_gray, reference);

    cv::namedWindow("draw", cv::WINDOW_AUTOSIZE);
    cv::imshow("draw", target);
    cv::setMouseCallback("draw", mouseCallback, (void *)&target);
    mat_input_reference = reference.clone();
    mat_input_confidence = 0*cv::Mat::ones(mat_gray.size(),mat_gray.type());
    // mat_input_confidence = mat_gray;
        int show_count = 0;
    while (1)
    {
    		mouse_x = glb_mouse_x;
    		mouse_y = glb_mouse_y;
    		mouse_click = glb_mouse_click;
    		mouse_left = glb_mouse_left;


    		if (mouse_click)
        {
    			  draw_Trajectory_Byreference(&target);

            if(show_count%10==0)
            {
            cv::Mat target_temp;
            filtering_time = (double)getTickCount();
            cv::cvtColor(target, target_temp, cv::COLOR_BGR2YCrCb);

            std::vector<cv::Mat> src_channels;
            std::vector<cv::Mat> dst_channels;

            cv::split(target_temp,src_channels);

            cv::Mat result1 = cv::Mat(input.size(),input.type());
            cv::Mat result2 = cv::Mat(input.size(),input.type());

            dst_channels.push_back(src_channels[0]);
            cv::ximgproc::fastBilateralSolverFilter(input,src_channels[1],mat_input_confidence,result1,sigma_spatial,sigma_luma,sigma_chroma);
            dst_channels.push_back(result1);
            cv::ximgproc::fastBilateralSolverFilter(input,src_channels[2],mat_input_confidence,result2,sigma_spatial,sigma_luma,sigma_chroma);
            dst_channels.push_back(result2);

            cv::merge(dst_channels,target_temp);
            cv::cvtColor(target_temp, target_temp, cv::COLOR_YCrCb2BGR);
            filtering_time = ((double)getTickCount() - filtering_time)/getTickFrequency();
            std::cout << "solver time: " << filtering_time << "ms" << std::endl;

            			cv::imshow("draw", target_temp);
                  std::cout << "/* message */"<<show_count << '\n';
            }
            show_count++;
    		}
    		if (cv::waitKey(2) == 27)
    		    break;
    }
    cv::cvtColor(target, target, cv::COLOR_BGR2YCrCb);

    std::vector<cv::Mat> src_channels;
    std::vector<cv::Mat> dst_channels;

    cv::split(target,src_channels);

    cv::Mat result1 = cv::Mat(input.size(),input.type());
    cv::Mat result2 = cv::Mat(input.size(),input.type());

    filtering_time = (double)getTickCount();

    dst_channels.push_back(src_channels[0]);
    cv::ximgproc::fastBilateralSolverFilter(input,src_channels[1],mat_input_confidence,result1,sigma_spatial,sigma_luma,sigma_chroma);
    dst_channels.push_back(result1);
    cv::ximgproc::fastBilateralSolverFilter(input,src_channels[2],mat_input_confidence,result2,sigma_spatial,sigma_luma,sigma_chroma);
    dst_channels.push_back(result2);

    cv::merge(dst_channels,target);
    cv::cvtColor(target, target, cv::COLOR_YCrCb2BGR);

    filtering_time = ((double)getTickCount() - filtering_time)/getTickFrequency();
    std::cout << "solver time: " << filtering_time << "ms" << std::endl;



    // cv::equalizeHist(result, result);
	  cv::imshow("input",input);
  	cv::imshow("output",target);

	  cv::waitKey(0);


    return 0;
}
