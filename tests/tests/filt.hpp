
#ifndef _FILT_HPP_
#define _FILT_HPP_

#include <iostream>

#include "blur.hpp"
#include "unique.hpp"
#include "splat.hpp"
#include "slice.hpp"
#include "testslib.hpp"
#include "csrmatrix.hpp"
#include "hashcoords.hpp"
#include "getvalididx.hpp"
#include "factorization.hpp"

#include<opencv2/core/core.hpp>
// #include<opencv2/core/eigen.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/opencv.hpp>



    void filt(std::vector<float>& x, std::vector<float>& w,std::vector<float>& out)
    {

        for (int i = 0; i < npixels; i++)
        {
            x[i] = x[i] * w[i];
        }

        std::vector<float> spalt_result;
        std::vector<float> blur_result;
        std::vector<float> slice_result(npixels, -1);

        std::vector<float> spalt_wresult;
        std::vector<float> blur_wresult;
        std::vector<float> slice_wresult(npixels, -1);

        std::vector<float> onesx(npixels,1);
        std::vector<float> spalt_onesresult;
        std::vector<float> blur_onesresult;
        std::vector<float> slice_onesresult(npixels, -1);


        Splat(x, spalt_result);
        Splat(w, spalt_wresult);
        Splat(onesx, spalt_onesresult);
        // std::cout << "x:" << std::endl;
        // PrintVector(x);
        // std::cout << "spalt_result:" << std::endl;
        // PrintVector(spalt_result);

        Blur(spalt_result, blur_result);
        Blur(spalt_wresult, blur_wresult);
        Blur(spalt_onesresult, blur_onesresult);
        // std::cout << "blur_result:" << std::endl;
        // PrintVector(blur_result);

        Slice(blur_result, slice_result);
        Slice(blur_wresult, slice_wresult);
        Slice(blur_onesresult, slice_onesresult);
        // std::cout << "slice_result:" << std::endl;
        // PrintVector(slice_result);
        // std::cout << "slice_onesresult:" << std::endl;
        // PrintVector(slice_onesresult);

        out.resize(npixels);
        for (int i = 0; i < npixels; i++)
        {
            out[i] = slice_result[i]/slice_wresult[i];
            // out[i] = slice_result[i]/slice_onesresult[i];
            // out[i] = slice_wresult[i]/slice_onesresult[i];
        }


    }


    void test_filt()
    {

        std::cout << "hello filter" << '\n';

        clock_t now;
        now = clock();
        printf( "start : now is %f seconds\n\n", (float)(now) / CLOCKS_PER_SEC);

        cv::Mat reference = cv::imread("reference.png");
        cv::Mat im1 = cv::imread("reference.png");
        cv::Mat target = cv::imread("target.png");
        cv::Mat confidence = cv::imread("confidence.png");
        // cv::Mat target = cv::imread(argv[2],0);

    	cvtColor(reference, reference, cv::COLOR_BGR2YCrCb);
    	cvtColor(target, target, cv::COLOR_BGR2YCrCb);
    	cvtColor(confidence, confidence, cv::COLOR_BGR2YCrCb);

        std::cout << "reference:" << reference.cols<<"x"<< reference.rows<< std::endl;


        float spatialSigma = 32.0;
        float lumaSigma = 16.0;
        float chromaSigma = 16.0;

        npixels = reference.cols*reference.rows;

        std::vector<float> ref(reference.cols*reference.rows*5);
        std::vector<float> tar(reference.cols*reference.rows);
        std::vector<float> con(reference.cols*reference.rows);
        int idx = 0;

    	std::cout << "start filling positions and values" << std::endl;
        now = clock();
        printf( "filling : now is %f seconds\n", (float)(now) / CLOCKS_PER_SEC);
        for (int y = 0; y < reference.cols; y++) {
            for (int x = 0; x < reference.rows; x++) {
                ref[idx*5+0] = ceilf(x/spatialSigma);
                ref[idx*5+1] = ceilf(y/spatialSigma);
                ref[idx*5+2] = ceilf(reference.at<cv::Vec3b>(x,y)[0]/lumaSigma);
                ref[idx*5+3] = ceilf(reference.at<cv::Vec3b>(x,y)[1]/chromaSigma);
                ref[idx*5+4] = ceilf(reference.at<cv::Vec3b>(x,y)[2]/chromaSigma);
                tar[idx] = target.at<cv::Vec3b>(x,y)[0];
                con[idx] = confidence.at<cv::Vec3b>(x,y)[0];
                // values[idx*4+1] = target.at<uchar>(x,y);
                // values[idx*4+2] = target.at<uchar>(x,y);
                // values[idx*4+1] = 1.0f;
                // values[idx*4+2] = 1.0f;
                // values[idx*4+3] = 1.0f;
                idx++;
            }
        }


        compute_factorization(ref);

        now = clock();
        printf( "filt : now is %f seconds\n\n", (float)(now) / CLOCKS_PER_SEC);
        filt(tar,con,tar);
        now = clock();
        printf( "filted : now is %f seconds\n\n", (float)(now) / CLOCKS_PER_SEC);

        // Divide through by the homogeneous coordinate and store the
        // result back to the image
        idx = 0;
        for (int y = 0; y < reference.cols; y++) {
            for (int x = 0; x < reference.rows; x++) {
                // float w = values[idx*4+3];
                target.at<cv::Vec3b>(x,y)[0] = tar[idx];
                // target.at<cv::uchar>(x,y) = values[idx*4+1]/w;
                // target.at<cv::uchar>(x,y) = values[idx*4+2]/w;
                // target.at<cv::Vec3b>(x,y)[0] = values[idx*4+0]/w;
                // target.at<cv::Vec3b>(x,y)[1] = values[idx*4+0]/w;
                // target.at<cv::Vec3b>(x,y)[2] = values[idx*4+0]/w;
                idx++;
            }
        }

    	cvtColor(reference, reference, cv::COLOR_YCrCb2BGR);
    	cvtColor(target, target, cv::COLOR_YCrCb2BGR);


        now = clock();
        printf( "finished : now is %f seconds\n\n", (float)(now) / CLOCKS_PER_SEC);
    	cv::imshow("input",im1);
    	cv::imshow("output",target);
    	cv::waitKey(0);



    }



#endif //_FILT_HPP_
