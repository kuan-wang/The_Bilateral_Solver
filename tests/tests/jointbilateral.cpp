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

#include <opencv2/ximgproc.hpp>

#include<opencv2/core/core.hpp>
// #include<opencv2/core/eigen.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/opencv.hpp>


	// #include <Eigen/Dense>
	// #include <Eigen/SparseCore>
	// #include <Eigen/SparseCholesky>
	// #include <Eigen/IterativeLinearSolvers>
	// #include <Eigen/Sparse>


//#include "stdafx.h"
//#include <opencv2/opencv.hpp>
//#include <ximgproc.hpp>

int main(int argc, char const *argv[])
{
	cv::Mat src = cv::imread(argv[1], 1); // 原始带噪声的深度图
	cv::Mat joint = cv::imread(argv[2], 1);
	cv::Mat dst;

        double colorSigma = double(atof(argv[3]));
        double spatialSigma = double(atof(argv[4]));

	int64 begin = cvGetTickCount();
	cv::ximgproc::jointBilateralFilter(joint, src, dst, -1, colorSigma, spatialSigma);
	int64 end = cvGetTickCount();

	float time = (end - begin) / (cvGetTickFrequency() * 1000.);
	printf("time = %fms\n", time);

	imshow("src", src);
	imshow("joint", joint);
	imshow("jointBilateralFilter", dst);
	cv::waitKey(0);


    return 0;
}
