
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>

#include<opencv2/core/core.hpp>
#include<opencv2/core/eigen.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/opencv.hpp>

#include <set>
#include <cmath>
#include <math.h>
#include <time.h>
#include <chrono>
#include <vector>
#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <opencv2/ximgproc.hpp>

#include "BilateralSolver.hpp"


void process(cv::Mat reference,cv::Mat target,float spatialSigma = 8,float lumaSigma = 8,float chromaSigma = 8,float fgs_spatialSigma = 2000,float fgs_colorSigma = 1.5);

void int2str(const int &int_temp,std::string &string_temp);

int main(int argc, char const *argv[]) {

        std::cout << "hello solver" << '\n';

        cv::Mat reference = cv::imread(argv[1],1);
        cv::Mat target = cv::imread(argv[2],0);
        // cv::Mat confidence = cv::imread(argv[3],0);
        float spatialSigma = float(atof(argv[4]));      //8.0
        float lumaSigma = float(atof(argv[5]));         //8.0
        float chromaSigma = float(atof(argv[6]));       //8.0
        float fgs_spatialSigma = float(atof(argv[7]));  //2000
        float fgs_colorSigma = float(atof(argv[8]));    //1.5

        std::string rgbd_file = argv[3];

        // process(reference,target,spatialSigma,lumaSigma,chromaSigma,fgs_spatialSigma,fgs_colorSigma);

        for (int i = 0; i < 10; i++) {
            std::string idx;
            int2str(i,idx);
            std::string reference_file = rgbd_file + "/rgb00000" + idx +".png";
            std::string target_file = rgbd_file + "/depth00000" + idx +".png";
            std::cout << "depth :"<< target_file << '\n';
            std::cout << "rgb :"<< reference_file << '\n';

            cv::Mat reference = cv::imread(reference_file,1);
            cv::Mat target = cv::imread(target_file,0);
            process(reference,target,spatialSigma,lumaSigma,chromaSigma,fgs_spatialSigma,fgs_colorSigma);
        }

        for (int i = 10; i < 100; i++) {
            std::string idx;
            int2str(i,idx);
            std::string reference_file = rgbd_file + "/rgb0000" + idx +".png";
            std::string target_file = rgbd_file + "/depth0000" + idx +".png";
            std::cout << "depth :"<< target_file << '\n';
            std::cout << "rgb :"<< reference_file << '\n';

            cv::Mat reference = cv::imread(reference_file,1);
            cv::Mat target = cv::imread(target_file,0);
            process(reference,target,spatialSigma,lumaSigma,chromaSigma,fgs_spatialSigma,fgs_colorSigma);
        }

        for (int i = 100; i < 950; i++) {
            std::string idx;
            int2str(i,idx);
            std::string reference_file = rgbd_file + "/rgb000" + idx +".png";
            std::string target_file = rgbd_file + "/depth000" + idx +".png";
            std::cout << "depth :"<< target_file << '\n';
            std::cout << "rgb :"<< reference_file << '\n';

            cv::Mat reference = cv::imread(reference_file,1);
            cv::Mat target = cv::imread(target_file,0);
            process(reference,target,spatialSigma,lumaSigma,chromaSigma,fgs_spatialSigma,fgs_colorSigma);
        }



  return 0;
}

void process(cv::Mat reference,cv::Mat target,float spatialSigma,float lumaSigma,float chromaSigma,float fgs_spatialSigma,float fgs_colorSigma)
{
      float filtering_time;

        clock_t now;
        now = clock();
        printf( "start : now is %f seconds\n\n", (float)(now) / CLOCKS_PER_SEC);

        cv::Mat confidence = 0*cv::Mat::ones(target.size(), CV_8UC1);
        for(int i = 0; i < target.rows;i++)
        {
            for(int j = 0; j < target.cols;j++)
            {
                if(target.at<uchar>(i,j) != 0 && reference.at<cv::Vec3b>(i,j)[0] != 0) confidence.at<uchar>(i,j) = 255;
            }
        }

        for(int i = 0; i < target.rows;i++)
        {
            for(int j = 0; j < target.cols;j++)
            {
                if(target.at<uchar>(i,j) != 0 && reference.at<cv::Vec3b>(i,j)[0] != 0) target.at<uchar>(i,j) = 255-target.at<uchar>(i,j);
            }
        }



        filtering_time = (float)cv::getTickCount();

        cv::equalizeHist(target, target);


#define ENABLE_FGS_FILTER
#ifdef ENABLE_FGS_FILTER
      	std::chrono::steady_clock::time_point start_fgs = std::chrono::steady_clock::now();
        cv::Mat x;
        cv::Mat w;
        cv::Mat xw;
        cv::Mat filtered_xw;
        cv::Mat filtered_w;
        cv::Mat filtered_disp;
	      target.convertTo(x, CV_32FC1, 1.0f/255.0f);
	      confidence.convertTo(w, CV_32FC1);
        xw = x.mul(w);
        cv::ximgproc::fastGlobalSmootherFilter(reference, xw, filtered_xw, fgs_spatialSigma, fgs_colorSigma);
        cv::ximgproc::fastGlobalSmootherFilter(reference, w, filtered_w, fgs_spatialSigma, fgs_colorSigma);
        cv::divide(filtered_xw, filtered_w, filtered_disp, 255.0f, CV_8UC1);

        // cv::ximgproc::fastGlobalSmootherFilter(reference, target, filtered_disp, spatialSigma*spatialSigma, lumaSigma);

      	std::chrono::steady_clock::time_point end_fgs = std::chrono::steady_clock::now();
      	std::cout << "fastGlobalSmootherFilter time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_fgs - start_fgs).count() << "ms" << std::endl;
        // cv::equalizeHist(filtered_disp, filtered_disp);
        cv::imshow("fgs_filtered_disp",filtered_disp);

#endif //ENABLE_FGS_FILTER





    	  // cvtColor(reference, reference, cv::COLOR_BGR2YUV);
        std::cout << "reference:" << reference.cols<<"x"<< reference.rows<< std::endl;


      	std::chrono::steady_clock::time_point start_solver = std::chrono::steady_clock::now();
        cv::Mat result;
        // cv::ximgproc::fastBilateralSolverFilter(reference,filtered_disp,confidence,result,spatialSigma,lumaSigma,chromaSigma);
        cv::ximgproc::fastBilateralSolverFilter(reference,target,confidence,result,spatialSigma,lumaSigma,chromaSigma,100);
      	std::chrono::steady_clock::time_point end_solver = std::chrono::steady_clock::now();
      	std::cout << "solver time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_solver - start_solver).count() << "ms" << std::endl;


      cv::equalizeHist(result, result);
      // cv::normalize(result,result,255,0,cv::NORM_MINMAX);
    	cv::imshow("input",reference);
    	cv::imshow("target",target);
    	cv::imshow("output",result);

// #define ENABLE_DOMAIN_TRANSFORM_FILTER
#ifdef ENABLE_DOMAIN_TRANSFORM_FILTER
	const float property_dt_sigmaSpatial = 40.0f;
	const float property_dt_sigmaColor = 220.0f;
	const int property_dt_numIters = 3;
	cv::Mat final_disparty_dtfiltered_image;
	cv::ximgproc::dtFilter(reference,
		// filtered_disp, final_disparty_dtfiltered_image,
		result, final_disparty_dtfiltered_image,
		property_dt_sigmaSpatial, property_dt_sigmaColor,
		cv::ximgproc::DTF_RF,
		property_dt_numIters);

	// display disparity image
	cv::Mat adjmap_dt;
	final_disparty_dtfiltered_image.convertTo(adjmap_dt, CV_8UC1);
		// 255.0f / 255.0f, 0.0f);
  // cv::imwrite("output+df.png",adjmap_dt);
	cv::imshow("disparity image + domain transform", adjmap_dt);
#endif
        filtering_time = ((float)cv::getTickCount() - filtering_time)/cv::getTickFrequency();
        std::cout<<"Filtering time: "<<filtering_time<<"s"<<std::endl;

    	cv::waitKey(1);

}
void int2str(const int &int_temp,std::string &string_temp)
{
        std::stringstream stream;
        stream<<int_temp;
        string_temp=stream.str();
}
