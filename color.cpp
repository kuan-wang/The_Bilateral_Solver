



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


        double spatialSigma = atof(argv[3]);
        double lumaSigma = atof(argv[4]);

	    bilateral(reference, target, spatialSigma, lumaSigma);


        now = clock();
        printf( "finished :now is %f seconds\n\n", (double)(now) / CLOCKS_PER_SEC);
    	cv::imshow("input",im1);
    	cv::imshow("output",target);
    	cv::waitKey(0);


        return 0;
    }
