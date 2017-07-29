
#ifndef _FILT_HPP_
#define _FILT_HPP_

#define ENABLE_DOMAIN_TRANSFORM_FILTER

#include <iostream>
#include <chrono>

#include "blur.hpp"
#include "unique.hpp"
#include "splat.hpp"
#include "slice.hpp"
#include "testslib.hpp"
#include "csrmatrix.hpp"
#include "hashcoords.hpp"
#include "bistochastize.hpp"
#include "getvalididx.hpp"
#include "factorization.hpp"

#include<opencv2/core/core.hpp>
#include<opencv2/core/eigen.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>

using namespace cv;

    void solve(Eigen::MatrixXd& x,
               Eigen::MatrixXd& w,
               Eigen::MatrixXd& out)
    {



      	std::chrono::steady_clock::time_point start_solve = std::chrono::steady_clock::now();

        // SparseMatrix<double> A_diag(nvertices);
        Eigen::SparseMatrix<double> M(nvertices,nvertices);
        Eigen::SparseMatrix<double> A_data(nvertices,nvertices);
        Eigen::SparseMatrix<double> A(nvertices,nvertices);
        Eigen::VectorXd b(nvertices);
        Eigen::VectorXd y(nvertices);
        Eigen::VectorXd w_splat(nvertices);
        Eigen::VectorXd xw(x.size());

      	std::chrono::steady_clock::time_point start_A_construction = std::chrono::steady_clock::now();
      	std::cout << "before A construction: " << std::chrono::duration_cast<std::chrono::milliseconds>(start_A_construction - start_solve).count() << "ms" << std::endl;

	    // std::cout << "start Splat(w,w_splat)" << std::endl;
        Splat(w,w_splat);
        diags(w_splat,A_data);
        A = bs_param.lam * (Dm - Dn * (blurs_test*Dn)) + A_data ;
      	std::chrono::steady_clock::time_point end_A_construction = std::chrono::steady_clock::now();
      	std::cout << "A construction: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_A_construction - start_A_construction).count() << "ms" << std::endl;

        xw = x.array() * w.array();
        Splat(xw,b);
      	std::chrono::steady_clock::time_point end_b_construction = std::chrono::steady_clock::now();
      	std::cout << "b construction: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_b_construction - end_A_construction).count() << "ms" << std::endl;


        // fill A and b
        Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower|Eigen::Upper> cg;
        cg.compute(A);
        cg.setMaxIterations(bs_param.cg_maxiter);
        cg.setTolerance(bs_param.cg_tol);
        y = cg.solve(b);
        std::cout << "#iterations:     " << cg.iterations() << std::endl;
        std::cout << "estimated error: " << cg.error()      << std::endl;
        // for (size_t i = 0; i < bs_param.cg_maxiter; i++) {
        //     y = cg.solve(b);
        //     std::cout << "#iterations:     " << cg.iterations() << std::endl;
        //     std::cout << "estimated error: " << cg.error()      << std::endl;
        //     if(cg.error()  < bs_param.cg_tol) break;
        // }
      	std::chrono::steady_clock::time_point end_solve_y = std::chrono::steady_clock::now();
      	std::cout << "solve Ay=b: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_solve_y - end_b_construction).count() << "ms" << std::endl;

        // Slice(y,out);

        out = Eigen::SparseMatrix<double,Eigen::RowMajor>(S.transpose())*y;
        // std::cout << out << std::endl;
      	std::chrono::steady_clock::time_point end_slice = std::chrono::steady_clock::now();
      	std::cout << "slice: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_slice - end_solve_y).count() << "ms" << std::endl;



    }




    void test_solve()
    {

        std::cout << "hello solver" << '\n';

        double filtering_time;

        clock_t now;
        now = clock();
        printf( "start : now is %f seconds\n\n", (double)(now) / CLOCKS_PER_SEC);

        // cv::Mat reference = cv::imread("reference.png",-1);
        // cv::Mat im1 = cv::imread("reference.png",-1);
        // cv::Mat target = cv::imread("target.png",-1);
        // cv::Mat confidence = cv::imread("confidence.png",-1);

        // cv::Mat reference = cv::imread("rgb.png",-1);
        // cv::Mat im1 = cv::imread("rgb.png",-1);
        // cv::Mat target = cv::imread("depth.png",-1);
        // cv::Mat confidence = cv::imread("depth.png",-1);

        cv::Mat reference = cv::imread(args[1],1);
        cv::Mat im1 = cv::imread(args[1],1);
        cv::Mat target = cv::imread(args[2],0);
        cv::Mat confidence = cv::imread(args[3],0);

        // cv::Mat reference = cv::imread("testr.png",-1);
        // cv::Mat im1 = cv::imread("testr.png",-1);
        // // cv::Mat reference(im1.rows,im1.cols,CV_32FC3);
        // cv::Mat target = cv::imread("testt.png",-1);
        // cv::Mat confidence = cv::imread("testc.png",-1);

        // std::cout << reference << std::endl;
        // std::cout << target << std::endl;
        // std::cout << confidence << std::endl;

        // cv::Mat reference = cv::imread("rgb.png");
        // cv::Mat im1 = cv::imread("rgb.png");
        // cv::Mat target = cv::imread("depth.png");
        // cv::Mat confidence = cv::imread("depth.png");
        // cv::Mat target = cv::imread(argv[2],0);

    	cvtColor(reference, reference, cv::COLOR_BGR2YCrCb);
    	// cvtColor(target, target, cv::COLOR_BGR2YCrCb);
    	// cvtColor(confidence, confidence, cv::COLOR_BGR2YCrCb);

        std::cout << "reference:" << reference.cols<<"x"<< reference.rows<< std::endl;
        // std::cout << reference << std::endl;


        double spatialSigma = double(atof(args[4]));
        double lumaSigma = double(atof(args[5]));
        double chromaSigma = double(atof(args[6]));

        npixels = reference.cols*reference.rows;

        cv::Mat r(npixels, 5, CV_8U);
        cv::Mat t(npixels, 1, CV_64F);
        cv::Mat c(npixels, 1, CV_64F);
        // std::vector<double> re(reference.cols*reference.rows*5);
        // std::vector<double> ta(reference.cols*reference.rows);
        // std::vector<double> co(reference.cols*reference.rows);
        int idx = 0;
    	std::cout << "start filling positions and values" << std::endl;
        now = clock();
        printf( "fill positions and values : now is %f seconds\n", (double)(now) / CLOCKS_PER_SEC);
        for (int y = 0; y < reference.rows; y++) {
            for (int x = 0; x < reference.cols; x++) {
                uchar *datar = r.ptr<uchar>(idx);
                datar[0] = int(x/spatialSigma);
                datar[1] = int(y/spatialSigma);
                datar[2] = int(reference.at<cv::Vec3b>(y,x)[2]/lumaSigma);
                datar[3] = int((reference.at<cv::Vec3b>(y,x)[1])/chromaSigma);
                datar[4] = int((reference.at<cv::Vec3b>(y,x)[0])/chromaSigma);
                // datar[0] = floorf(x/spatialSigma);
                // datar[1] = floorf(y/spatialSigma);
                // datar[2] = floorf(reference.at<cv::Vec3b>(y,x)[0]/lumaSigma);
                // datar[3] = floorf(reference.at<cv::Vec3b>(y,x)[1]/chromaSigma);
                // datar[4] = floorf(reference.at<cv::Vec3b>(y,x)[2]/chromaSigma);
                // datar[3] = 1.0;
                // datar[4] = 1.0;
                idx++;
            }
        }
        idx = 0;
        for (int y = 0; y < reference.rows; y++) {
            for (int x = 0; x < reference.cols; x++) {
                double *datac = c.ptr<double>(idx);
                // datac[0] = 1.0;
                datac[0] = double(confidence.at<uchar>(y,x))/255.0;
                // if(datac[0] != 0) datac[0] = 0.1/datac[0];
                // datac[0] = confidence.at<double>(x,y)[0];
                idx++;
            }
        }
        idx = 0;
        for (int y = 0; y < reference.rows; y++) {
            for (int x = 0; x < reference.cols; x++) {
                double *datat = t.ptr<double>(idx);
                datat[0] = double(target.at<uchar>(y,x))/255.0;
                // datat[0] = target.at<cv::Vec3b>(x,y)[0];
                idx++;
            }
        }

        std::cout << "cv2eigen" << std::endl;
        Eigen::Matrix<uchar, Eigen::Dynamic, Eigen::Dynamic> ref_temp;
        Eigen::Matrix<long long, Eigen::Dynamic, Eigen::Dynamic> ref;
        Eigen::MatrixXd tar;
        Eigen::MatrixXd con;

        cv::cv2eigen(r,ref_temp);
        cv::cv2eigen(t,tar);
        cv::cv2eigen(c,con);
        ref = ref_temp.cast<long long>();
        std::cout << "finished cv2eigen" << std::endl;


    	// std::cout << "start filling positions and values" << std::endl;
        // now = clock();
        // printf( "fill positions and values : now is %f seconds\n", (double)(now) / CLOCKS_PER_SEC);
        // idx = 0;
        // for (int y = 0; y < reference.cols; y++) {
        //     for (int x = 0; x < reference.rows; x++) {
        //         re[idx*5+0] = ceilf(x/spatialSigma);
        //         re[idx*5+1] = ceilf(y/spatialSigma);
        //         re[idx*5+2] = ceilf(reference.at<cv::Vec3b>(x,y)[0]/lumaSigma);
        //         re[idx*5+3] = ceilf(reference.at<cv::Vec3b>(x,y)[1]/chromaSigma);
        //         re[idx*5+4] = ceilf(reference.at<cv::Vec3b>(x,y)[2]/chromaSigma);
        //
        //         // ref[idx*5+0] = (x/spatialSigma);
        //         // ref[idx*5+1] = (y/spatialSigma);
        //         // ref[idx*5+2] = (reference.at<cv::Vec3b>(x,y)[0]/lumaSigma);
        //         // ref[idx*5+3] = (reference.at<cv::Vec3b>(x,y)[1]/chromaSigma);
        //         // ref[idx*5+4] = (reference.at<cv::Vec3b>(x,y)[2]/chromaSigma);
        //         // tar[idx] = target.at<cv::Vec3b>(x,y)[0];
        //         // con[idx] = confidence.at<cv::Vec3b>(x,y)[0];
        //         // values[idx*4+1] = target.at<uchar>(x,y);
        //         // values[idx*4+2] = target.at<uchar>(x,y);
        //         // values[idx*4+1] = 1.0f;
        //         // values[idx*4+2] = 1.0f;
        //         // values[idx*4+3] = 1.0f;
        //         idx++;
        //     }
        // }

        filtering_time = (double)getTickCount();

      	std::chrono::steady_clock::time_point start_solver = std::chrono::steady_clock::now();
        compute_factorization(reference, spatialSigma, lumaSigma, chromaSigma);
        // compute_factorization(ref);
        // compute_factorization(re);
        // std::cout << ref << std::endl;

        // bistochastize();

        solve(tar,con,tar);

      	std::chrono::steady_clock::time_point end_solver = std::chrono::steady_clock::now();
      	std::cout << "solver time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_solver - start_solver).count() << "ms" << std::endl;
        // Divide through by the homogeneous coordinate and store the
        // result back to the image
        idx = 0;
        for (int y = 0; y < reference.rows; y++) {
            for (int x = 0; x < reference.cols; x++) {
                // double w = values[idx*4+3];
                target.at<uchar>(y,x) = tar(idx)*255.0;
                // target.at<cv::uchar>(x,y) = values[idx*4+1]/w;
                // target.at<cv::uchar>(x,y) = values[idx*4+2]/w;
                // target.at<cv::Vec3b>(x,y)[0] = values[idx*4+0]/w;
                // target.at<cv::Vec3b>(x,y)[1] = values[idx*4+0]/w;
                // target.at<cv::Vec3b>(x,y)[2] = values[idx*4+0]/w;
                idx++;
            }
        }

    	// cvtColor(reference, reference, cv::COLOR_YCrCb2BGR);
    	// cvtColor(target, target, cv::COLOR_YCrCb2BGR);

      cv::equalizeHist(target, target);
    	cv::imshow("input",im1);
    	cv::imshow("output",target);

#ifdef ENABLE_DOMAIN_TRANSFORM_FILTER
	const float property_dt_sigmaSpatial = 40.0f;
	const float property_dt_sigmaColor = 220.0f;
	const int property_dt_numIters = 3;
	cv::Mat final_disparty_dtfiltered_image;
	cv::ximgproc::dtFilter(reference,
		target, final_disparty_dtfiltered_image,
		property_dt_sigmaSpatial, property_dt_sigmaColor,
		cv::ximgproc::DTF_RF,
		property_dt_numIters);

	// display disparity image
	cv::Mat adjmap_dt;
	final_disparty_dtfiltered_image.convertTo(adjmap_dt, CV_8UC1);
		// 255.0f / 255.0f, 0.0f);
	cv::imshow("disparity image + domain transform", adjmap_dt);
#endif
        filtering_time = ((double)getTickCount() - filtering_time)/getTickFrequency();
        cout<<"Filtering time: "<<filtering_time<<"s"<<endl;

    	cv::waitKey(0);



    }



#endif //_FILT_HPP_
