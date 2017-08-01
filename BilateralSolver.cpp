
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
#include <iostream>
#include <iterator>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>

#include "BilateralSolver.hpp"

namespace cv
{
namespace ximgproc
{


    void FastBilateralSolverFilter::init(cv::Mat& reference_bgr, float sigma_spatial, float sigma_luma, float sigma_chroma)
    {

	      cv::Mat reference_yuv;
	      cv::cvtColor(reference_bgr, reference_yuv, CV_BGR2YUV);

	      std::chrono::steady_clock::time_point begin_grid_construction = std::chrono::steady_clock::now();

	      const int w = reference_yuv.cols;
	      const int h = reference_yuv.rows;
        npixels = w*h;
	      std::int64_t hash_vec[5];
	      for (int i = 0; i < 5; ++i)
		        hash_vec[i] = static_cast<std::int64_t>(std::pow(255, i));

	      std::unordered_map<std::int64_t /* hash */, int /* vert id */> hashed_coords;
	      hashed_coords.reserve(w*h);

	      const unsigned char* pref = (const unsigned char*)reference_yuv.data;
	      int vert_idx = 0;
	      int pix_idx = 0;


      	// construct Splat(Slice) matrices
        splat_idx.resize(npixels);
    	  for (int y = 0; y < h; ++y)
      	{
    	    for (int x = 0; x < w; ++x)
      		{
      			std::int64_t coord[5];
      			coord[0] = int(x / sigma_spatial);
      			coord[1] = int(y / sigma_spatial);
      			coord[2] = int(pref[0] / sigma_luma);
      			coord[3] = int(pref[1] / sigma_chroma);
      			coord[4] = int(pref[2] / sigma_chroma);

      			// convert the coordinate to a hash value
      			std::int64_t hash_coord = 0;
      			for (int i = 0; i < 5; ++i)
      				  hash_coord += coord[i] * hash_vec[i];

      			// pixels whom are alike will have the same hash value.
      			// We only want to keep a unique list of hash values, therefore make sure we only insert
      			// unique hash values.
      			auto it = hashed_coords.find(hash_coord);
      			if (it == hashed_coords.end())
      			{
      				hashed_coords.insert(std::pair<std::int64_t, int>(hash_coord, vert_idx));
              splat_idx[pix_idx] = vert_idx;
      				++vert_idx;
      			}
      			else
      			{
              splat_idx[pix_idx] = it->second;
      			}

      			pref += 3; // skip 3 bytes (y u v)
      			++pix_idx;
      		}
      	}
        nvertices = hashed_coords.size();


      	// construct Blur matrices
      	std::chrono::steady_clock::time_point begin_blur_construction = std::chrono::steady_clock::now();
        Eigen::VectorXf ones_nvertices = Eigen::VectorXf::Ones(nvertices);
        Eigen::VectorXf ones_npixels = Eigen::VectorXf::Ones(npixels);
        blurs_test = ones_nvertices.asDiagonal();
        blurs_test *= 10;
        for(int offset = -1; offset <= 1;++offset)
        {
            if(offset == 0) continue;
          	for (int i = 0; i < 5; ++i)
          	{
          	     Eigen::SparseMatrix<float, Eigen::ColMajor> blur_temp(hashed_coords.size(), hashed_coords.size());
                 blur_temp.reserve(Eigen::VectorXi::Constant(nvertices,6));
          		   std::int64_t offset_hash_coord = offset * hash_vec[i];
        		     for (auto it = hashed_coords.begin(); it != hashed_coords.end(); ++it)
      		       {
      			         std::int64_t neighb_coord = it->first + offset_hash_coord;
      			         auto it_neighb = hashed_coords.find(neighb_coord);
      			         if (it_neighb != hashed_coords.end())
          			     {
                         blur_temp.insert(it->second,it_neighb->second) = 1.0f;
                         blur_idx.push_back(std::pair<int,int>(it->second, it_neighb->second));
          			     }
          		   }
                 blurs_test += blur_temp;
              }
        }
        blurs_test.finalize();


        //bistochastize
        int maxiter = 10;
        n = ones_nvertices;
        m = Eigen::VectorXf::Zero(nvertices);
        for (int i = 0; i < splat_idx.size(); i++) {
            m(splat_idx[i]) += 1.0f;
        }

        Eigen::VectorXf bluredn(nvertices);

        for (int i = 0; i < maxiter; i++) {
            Blur(n,bluredn);
            n = ((n.array()*m.array()).array()/bluredn.array()).array().sqrt();
        }
        Blur(n,bluredn);

        m = n.array() * (bluredn).array();
        Dm = m.asDiagonal();
        Dn = n.asDiagonal();
    }

    void FastBilateralSolverFilter::Splat(Eigen::VectorXf& input, Eigen::VectorXf& output)
    {
        output.setZero();
        for (int i = 0; i < splat_idx.size(); i++) {
            output(splat_idx[i]) += input(i);
        }

    }

    void FastBilateralSolverFilter::Blur(Eigen::VectorXf& input, Eigen::VectorXf& output)
    {
        output.setZero();
        output = input * 10;
        for (int i = 0; i < blur_idx.size(); i++)
        {
            output(blur_idx[i].first) += input(blur_idx[i].second);
        }
    }


    void FastBilateralSolverFilter::Slice(Eigen::VectorXf& input, Eigen::VectorXf& output)
    {
        output.setZero();
        for (int i = 0; i < splat_idx.size(); i++) {
            output(i) = input(splat_idx[i]);
        }
    }

    void FastBilateralSolverFilter::filt(cv::Mat& target,
                cv::Mat& confidence,
                cv::Mat& output)
    {

        Eigen::VectorXf x(npixels);
        Eigen::VectorXf w(npixels);
        Eigen::VectorXf xw(npixels);
        Eigen::VectorXf splat_xw(nvertices);
        Eigen::VectorXf splat_w(nvertices);
        Eigen::VectorXf blur_xw(nvertices);
        Eigen::VectorXf blur_w(nvertices);
        Eigen::VectorXf slice_xw(npixels);
        Eigen::VectorXf slice_w(npixels);


      	const uchar *pft = reinterpret_cast<const uchar*>(target.data);
      	for (int i = 0; i < npixels; i++)
      	{
      		  x(i) = float(pft[i])/255.0f;
        }
      	const uchar *pfc = reinterpret_cast<const uchar*>(confidence.data);
      	for (int i = 0; i < npixels; i++)
      	{
      		  w(i) = float(pfc[i])/255.0f;
        }
        // xw.setZero();
        for (int i = 0; i < x.size(); i++) {
            xw(i) = x(i) * w(i);
        }

        //splat
        Splat(xw, splat_xw);
        Splat(w, splat_w);

        //blur
        Blur(splat_xw, blur_xw);
        Blur(splat_w, blur_w);

        //slice
      	uchar *pftar = (uchar*)(output.data);
      	for (int i = 0; i < splat_idx.size(); i++)
      	{
      		  pftar[i] = (blur_xw(splat_idx[i])/blur_w(splat_idx[i])) * 255;
        }
    }



    void FastBilateralSolverFilter::solve(cv::Mat& target,
               cv::Mat& confidence,
               cv::Mat& output)
    {

        Eigen::VectorXf x(npixels);
        Eigen::VectorXf w(npixels);

      	const uchar *pft = reinterpret_cast<const uchar*>(target.data);
      	for (int i = 0; i < npixels; i++)
      	{
      		  x(i) = float(pft[i])/255.0f;
        }
      	const uchar *pfc = reinterpret_cast<const uchar*>(confidence.data);
      	for (int i = 0; i < npixels; i++)
      	{
      		  w(i) = float(pfc[i])/255.0f;
        }

        Eigen::SparseMatrix<float, Eigen::ColMajor> M(nvertices,nvertices);
        Eigen::SparseMatrix<float, Eigen::ColMajor> A_data(nvertices,nvertices);
        Eigen::SparseMatrix<float, Eigen::ColMajor> A(nvertices,nvertices);
        Eigen::VectorXf b(nvertices);
        Eigen::VectorXf y(nvertices);
        Eigen::VectorXf w_splat(nvertices);
        Eigen::VectorXf xw(x.size());



        //construct A
        Splat(w,w_splat);
        A_data = (w_splat).asDiagonal();
        A = bs_param.lam * (Dm - Dn * (blurs_test*Dn)) + A_data ;

        //construct b
        b.setZero();
        for (int i = 0; i < splat_idx.size(); i++) {
            b(splat_idx[i]) += x(i) * w(i);
        }

        // solve Ay = b
        Eigen::ConjugateGradient<Eigen::SparseMatrix<float>, Eigen::Lower|Eigen::Upper> cg;
        cg.compute(A);
        cg.setMaxIterations(bs_param.cg_maxiter);
        cg.setTolerance(bs_param.cg_tol);
        y = cg.solve(b);
        // std::cout << "#iterations:     " << cg.iterations() << std::endl;
        // std::cout << "estimated error: " << cg.error()      << std::endl;

        //slice
      	uchar *pftar = (uchar*)(output.data);
      	for (int i = 0; i < splat_idx.size(); i++)
      	{
      		  pftar[i] = y(splat_idx[i]) * 255;
        }


    }


}
}


int main(int argc, char const *argv[]) {

        std::cout << "hello solver" << '\n';

        float filtering_time;

        clock_t now;
        now = clock();
        printf( "start : now is %f seconds\n\n", (float)(now) / CLOCKS_PER_SEC);

        cv::Mat reference = cv::imread(argv[1],1);
        cv::Mat im1 = cv::imread(argv[1],1);
        cv::Mat target = cv::imread(argv[2],0);
        cv::Mat confidence = cv::imread(argv[3],0);



        float spatialSigma = float(atof(argv[4]));
        float lumaSigma = float(atof(argv[5]));
        float chromaSigma = float(atof(argv[6]));
        float fgs_spatialSigma = float(atof(argv[7]));
        float fgs_colorSigma = float(atof(argv[8]));



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
        cv::equalizeHist(filtered_disp, filtered_disp);
        cv::imshow("fgs_filtered_disp",filtered_disp);






    	  cvtColor(reference, reference, cv::COLOR_BGR2YUV);
    	  // cvtColor(reference, reference, cv::COLOR_BGR2YCrCb);
        std::cout << "reference:" << reference.cols<<"x"<< reference.rows<< std::endl;

        filtering_time = (float)cv::getTickCount();

      	std::chrono::steady_clock::time_point start_solver = std::chrono::steady_clock::now();
        // init(reference, spatialSigma, lumaSigma, chromaSigma);

        // filter(target,confidence,target);
        // solve(target,confidence,target);
        // FastBilateralSolverFilter::filter(reference,target,confidence,target,spatialSigma,lumaSigma,chromaSigma);
        cv::Mat result;
        cv::ximgproc::FastBilateralSolverFilter::filter(reference,target,confidence,result,spatialSigma,lumaSigma,chromaSigma);

      	std::chrono::steady_clock::time_point end_solver = std::chrono::steady_clock::now();
      	std::cout << "solver time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_solver - start_solver).count() << "ms" << std::endl;


      cv::equalizeHist(result, result);
    	cv::imshow("input",im1);
    	cv::imshow("output",result);

#define ENABLE_DOMAIN_TRANSFORM_FILTER
#ifdef ENABLE_DOMAIN_TRANSFORM_FILTER
	const float property_dt_sigmaSpatial = 40.0f;
	const float property_dt_sigmaColor = 220.0f;
	const int property_dt_numIters = 3;
	cv::Mat final_disparty_dtfiltered_image;
	cv::ximgproc::dtFilter(reference,
		result, final_disparty_dtfiltered_image,
		property_dt_sigmaSpatial, property_dt_sigmaColor,
		cv::ximgproc::DTF_RF,
		property_dt_numIters);

	// display disparity image
	cv::Mat adjmap_dt;
	final_disparty_dtfiltered_image.convertTo(adjmap_dt, CV_8UC1);
		// 255.0f / 255.0f, 0.0f);
	cv::imshow("disparity image + domain transform", adjmap_dt);
#endif
        filtering_time = ((float)cv::getTickCount() - filtering_time)/cv::getTickFrequency();
        std::cout<<"Filtering time: "<<filtering_time<<"s"<<std::endl;

    	cv::waitKey(0);


  return 0;
}
