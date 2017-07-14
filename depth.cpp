



#include <iostream>

#include "BilateralSolver.hpp"

#include<opencv2/core/core.hpp>
// #include<opencv2/core/eigen.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/opencv.hpp>

// #include <Eigen/Dense>
// #include <Eigen/SparseCore>
// #include <Eigen/SparseCholesky>
// #include <Eigen/IterativeLinearSolvers>
// #include <Eigen/Sparse>









    int main(int argc, char const *argv[])
    {


        std::cout << "hello solver" << '\n';

        clock_t now;
        now = clock();
        printf( "start : now is %f seconds\n\n", (double)(now) / CLOCKS_PER_SEC);

        cv::Mat reference = cv::imread(argv[1]);
        cv::Mat im1 = cv::imread(argv[1]);
        cv::Mat target = cv::imread(argv[2]);
        cv::Mat confidence = cv::imread(argv[3]);

        // cv::Mat reference = cv::imread("reference.png");
        // cv::Mat im1 = cv::imread("reference.png");
        // cv::Mat target = cv::imread("target.png");
        // cv::Mat confidence = cv::imread("confidence.png");

        // cv::Mat reference = cv::imread("rgb.png");
        // cv::Mat im1 = cv::imread("rgb.png");
        // cv::Mat target = cv::imread("depth.png");
        // cv::Mat confidence = cv::imread("depth.png");
        // cv::Mat target = cv::imread(argv[2],0);

        double spatialSigma = atof(argv[4]);
        double lumaSigma = atof(argv[5]);
        double chromaSigma = atof(argv[6]);

	    bilateral(reference, target, confidence, spatialSigma, lumaSigma, chromaSigma);


        now = clock();
        printf( "finished :now is %f seconds\n\n", (double)(now) / CLOCKS_PER_SEC);
    	cv::imshow("input",im1);
    	cv::imshow("output",target);
    	cv::waitKey(0);


        return 0;
    }
