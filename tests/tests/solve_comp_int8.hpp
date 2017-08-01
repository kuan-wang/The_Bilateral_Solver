
#ifndef _FILT_HPP_
#define _FILT_HPP_

#define ENABLE_DOMAIN_TRANSFORM_FILTER

#include <iostream>
#include <chrono>
#include <cmath>

#include "testslib.hpp"
// #include "factorization.hpp"

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



    void Splat(Eigen::VectorXf& input, Eigen::VectorXf& output);
    void Blur(Eigen::VectorXf& input, Eigen::VectorXf& output);
    void Slice(Eigen::VectorXf& input, Eigen::VectorXf& output);


    void compute_factorization(cv::Mat& reference_bgr, float sigma_spatial, float sigma_luma, float sigma_chroma)
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


      	// loop through each pixel of the image
        // Eigen::SparseMatrix<float, Eigen::ColMajor> S_temp(npixels,npixels);
        // S_temp.reserve(Eigen::VectorXi::Constant(npixels,1));
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
              // S_temp.insert(vert_idx,pix_idx) = 1.0f;
              splat_idx[pix_idx] = vert_idx;
      				++vert_idx;
      			}
      			else
      			{
              // S_temp.insert(it->second,pix_idx) = 1.0f;
              splat_idx[pix_idx] = it->second;
      			}

      			pref += 3; // skip 3 bytes (y u v)
      			++pix_idx;
      		}
      	}
        nvertices = hashed_coords.size();
        // S_temp.makeCompressed();
        // S = Eigen::SparseMatrix<float, Eigen::ColMajor>(nvertices, npixels);
        // S = S_temp.middleRows(0, nvertices);
        // for (int i = 0; i < npixels; i++)
        // {
            // S.insert(splat_idx[i],i) = 1.0f;
        // }
        // S.finalize();
        std::cout << "nvertices:"<<nvertices << '\n';
        // std::cout << S_temp.nonZeros() << '\n';
        // std::cout << S_temp.rows()<<"x"<<S_temp.cols() << '\n';
        // std::cout << S.nonZeros() << '\n';
        // std::cout << S.rows()<<"x"<<S.cols() << '\n';
      	std::chrono::steady_clock::time_point end_splat_construction = std::chrono::steady_clock::now();
      	std::cout << "S construction: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_splat_construction - begin_grid_construction).count() << "ms" << std::endl;






      	// Blur matrices   // Eigen::ColMajor or Eigen::ColMajor???
      	std::chrono::steady_clock::time_point begin_blur_construction = std::chrono::steady_clock::now();
        // blurs_test = Eigen::SparseMatrix<float, Eigen::ColMajor>(nvertices,nvertices);
        Eigen::VectorXf ones_nvertices = Eigen::VectorXf::Ones(nvertices);
        Eigen::VectorXf ones_npixels = Eigen::VectorXf::Ones(npixels);
        blurs_test = ones_nvertices.asDiagonal();
        blurs_test *= 10;
        // for(int offset = 0; offset <= 1;++offset)
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
                //  blur_temp.makeCompressed();
                 blurs_test += blur_temp;
              }
        }

        blurs_test.finalize();
        std::cout <<"blurs_test:"<< blurs_test.nonZeros() << '\n';
        std::cout << blurs_test.rows()<<"x"<<blurs_test.cols() << '\n';
        // std::cout << S.nonZeros() << '\n';
        // std::cout << S.rows()<<"x"<<S.cols() << '\n';


      	std::chrono::steady_clock::time_point end_blur_construction = std::chrono::steady_clock::now();
      	std::cout << "blur construction: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_blur_construction - begin_blur_construction).count() << "ms" << std::endl;

        //bistochastize
        int maxiter = 10;
        n = ones_nvertices;
        // Eigen::VectorXf& n = ones_nvertices;

      	// std::chrono::steady_clock::time_point start_m_construction = std::chrono::steady_clock::now();
        // Eigen::VectorXf m = S*ones_npixels;
      	// std::chrono::steady_clock::time_point end_m_construction = std::chrono::steady_clock::now();
      	// std::cout << "m construction: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_m_construction - start_m_construction).count() << "ms" << std::endl;

        // Eigen::VectorXf m(nvertices);
        m = Eigen::VectorXf::Zero(nvertices);
        // m.setZero();   // it's better to make a copy of zeros and ones for multi_use?
        for (int i = 0; i < splat_idx.size(); i++) {
            m(splat_idx[i]) += 1.0f;
        }

        Eigen::VectorXf bluredn(nvertices);
        Eigen::VectorXf bluredntest(nvertices);

        for (int i = 0; i < maxiter; i++) {
            // bluredn = blurs_test*n;
            Blur(n,bluredn);
            n = ((n.array()*m.array()).array()/bluredn.array()).array().sqrt();
            // for (int i = 0; i < nvertices; i++)
            // {
                // n(i) = sqrt(n(i)*m(i)/bluredn(i));
            // }
        }

      	// std::chrono::steady_clock::time_point start_bluredn = std::chrono::steady_clock::now();
        // bluredn = blurs_test*n;
        Blur(n,bluredn);
      	// std::chrono::steady_clock::time_point end_bluredn = std::chrono::steady_clock::now();
      	// std::cout << "bluredn construction: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_bluredn - start_bluredn).count() << "ms" << std::endl;

      	// std::chrono::steady_clock::time_point start_bluredntest = std::chrono::steady_clock::now();
        // Blur(n,bluredntest);
      	// std::chrono::steady_clock::time_point end_bluredntest = std::chrono::steady_clock::now();
      	// std::cout << "testbluredn construction: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_bluredntest - start_bluredntest).count() << "ms" << std::endl;


        m = n.array() * (bluredn).array();
        // for (int i = 0; i < nvertices; i++) {
            // m(i) = n(i) * bluredn(i);
        // }
        // diags(m,Dm);
        // diags(n,Dn);
        Dm = m.asDiagonal();
        Dn = n.asDiagonal();
      	std::chrono::steady_clock::time_point end_bisto_construction = std::chrono::steady_clock::now();
      	std::cout << "Dm Dn construction: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_bisto_construction - end_blur_construction).count() << "ms" << std::endl;


    }

    void Splat(Eigen::VectorXf& input, Eigen::VectorXf& output)
    {
        output.setZero();
        for (int i = 0; i < splat_idx.size(); i++) {
            output(splat_idx[i]) += input(i);
        }

    }

    void Blur(Eigen::VectorXf& input, Eigen::VectorXf& output)
    {
        output.setZero();
        output = input * 10;
        for (int i = 0; i < blur_idx.size(); i++)
        {
            output(blur_idx[i].first) += input(blur_idx[i].second);
        }
    }


    void Slice(Eigen::VectorXf& input, Eigen::VectorXf& output)
    {
        output.setZero();
        for (int i = 0; i < splat_idx.size(); i++) {
            output(i) = input(splat_idx[i]);
        }

    }

    void filter(cv::Mat& target,
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
        Splat(xw, splat_xw);
        Splat(w, splat_w);
        Blur(splat_xw, blur_xw);
        Blur(splat_w, blur_w);
        // Slice(blur_xw, slice_xw);
        // Slice(blur_w, slice_w);
        // for (int i = 0; i < x.size(); i++) {
            // out(i) = slice_xw(i) / slice_w(i);
        // }

      	uchar *pftar = (uchar*)(output.data);
      	for (int i = 0; i < splat_idx.size(); i++)
      	{
      		  pftar[i] = (blur_xw(splat_idx[i])/blur_w(splat_idx[i])) * 255;
        }


    }

    //
    // void solve(Eigen::MatrixXf& x,
    //            Eigen::MatrixXf& w,
    //            Eigen::MatrixXf& out)
    // {
    // void solve(Eigen::VectorXf& x,
              //  Eigen::VectorXf& w,
              //  Eigen::VectorXf& out)
    void solve(cv::Mat& target,
               cv::Mat& confidence,
               cv::Mat& output)
    {
      	std::chrono::steady_clock::time_point start_solve = std::chrono::steady_clock::now();

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

        // SparseMatrix<float> A_diag(nvertices);
        Eigen::SparseMatrix<float, Eigen::ColMajor> M(nvertices,nvertices);
        Eigen::SparseMatrix<float, Eigen::ColMajor> A_data(nvertices,nvertices);
        Eigen::SparseMatrix<float, Eigen::ColMajor> A(nvertices,nvertices);
        Eigen::VectorXf b(nvertices);
        Eigen::VectorXf y(nvertices);
        Eigen::VectorXf w_splat(nvertices);
        Eigen::VectorXf xw(x.size());
        Eigen::VectorXf A_pre(nvertices);
        std::vector<float> A_vec(blur_idx.size() + nvertices);

      	std::chrono::steady_clock::time_point start_A_construction = std::chrono::steady_clock::now();
      	std::cout << "before A construction: " << std::chrono::duration_cast<std::chrono::milliseconds>(start_A_construction - start_solve).count() << "ms" << std::endl;

	    // std::cout << "start Splat(w,w_splat)" << std::endl;
      	// std::chrono::steady_clock::time_point start_ws_construction = std::chrono::steady_clock::now();
        // w_splat = S*w;
      	// std::chrono::steady_clock::time_point end_ws_construction = std::chrono::steady_clock::now();
      	// std::cout << "w_splat construction: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_ws_construction - start_ws_construction).count() << "ms" << std::endl;

      	std::chrono::steady_clock::time_point start_wst_construction = std::chrono::steady_clock::now();
        Splat(w,w_splat);
        // w_splat.setZero();
        // for (int i = 0; i < splat_idx.size(); i++) {
            // w_splat(splat_idx[i]) += w(i);
        // }
      	std::chrono::steady_clock::time_point end_wst_construction = std::chrono::steady_clock::now();
      	std::cout << "w_splat construction: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_wst_construction - start_wst_construction).count() << "ms" << std::endl;


        A_data = (w_splat).asDiagonal();
        // A.noalias() = bs_param.lam * (Dm - Dn * (blurs_test*Dn)) + A_data ;
        // A_data.setZero();
        // for (int i = 0; i < blur_idx.size(); i++) {
        //     // A_data.coeffRef(blur_idx[i].first,blur_idx[i].second) += n(blur_idx[i].first)*n(blur_idx[i].second);
        //     A_vec[i] = n(blur_idx[i].first)*n(blur_idx[i].second);
        // }
        // A_pre.setZero();
        // for (int i = 0; i < nvertices; i++) {
        //     A_pre(i) = bs_param.lam*(m(i) - n(i)*n(i)) + w_splat(i);
        // }
        // A = A_pre.asDiagonal();
        // A = A - bs_param.lam * A_data;
        A = bs_param.lam * (Dm - Dn * (blurs_test*Dn)) + A_data ;
      	std::chrono::steady_clock::time_point end_A_construction = std::chrono::steady_clock::now();
      	std::cout << "A construction: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_A_construction - start_A_construction).count() << "ms" << std::endl;

        // xw = x.array() * w.array();
        b.setZero();
        for (int i = 0; i < splat_idx.size(); i++) {
            b(splat_idx[i]) += x(i) * w(i);
        }
      	std::chrono::steady_clock::time_point end_b_construction = std::chrono::steady_clock::now();
      	std::cout << "b construction: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_b_construction - end_A_construction).count() << "ms" << std::endl;


        // fill A and b
        Eigen::ConjugateGradient<Eigen::SparseMatrix<float>, Eigen::Lower|Eigen::Upper> cg;
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

        // for (int i = 0; i < splat_idx.size(); i++) {
            // out(i) = y(splat_idx[i]);
        // }

      	uchar *pftar = (uchar*)(output.data);
      	for (int i = 0; i < splat_idx.size(); i++)
      	{
      		  pftar[i] = y(splat_idx[i]) * 255;
        }

        // Eigen::SparseMatrix<float, Eigen::ColMajor> Sli = (S.transpose());
        // out.noalias() = Sli*y;
        // out.noalias() = Eigen::SparseMatrix<float, Eigen::ColMajor>(S.transpose())*y;
        // out = Eigen::SparseMatrix<float, Eigen::ColMajor>(S.transpose())*y;
        // std::cout << out << std::endl;
      	std::chrono::steady_clock::time_point end_slice = std::chrono::steady_clock::now();
      	std::cout << "slice: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_slice - end_solve_y).count() << "ms" << std::endl;



    }




    void test_solve()
    {

        std::cout << "hello solver" << '\n';

        float filtering_time;

        clock_t now;
        now = clock();
        printf( "start : now is %f seconds\n\n", (float)(now) / CLOCKS_PER_SEC);

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



        float spatialSigma = float(atof(args[4]));
        float lumaSigma = float(atof(args[5]));
        float chromaSigma = float(atof(args[6]));
        float fgs_spatialSigma = float(atof(args[7]));
        float fgs_colorSigma = float(atof(args[8]));



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
        npixels = reference.cols*reference.rows;

        // Eigen::VectorXf tar(npixels);
        // Eigen::VectorXf con(npixels);
        //
      	// const uchar *pft = reinterpret_cast<const uchar*>(target.data);
      	// for (int i = 0; i < npixels; i++)
      	// {
      	// 	  tar(i) = float(pft[i])/255.0f;
        // }
      	// const uchar *pfc = reinterpret_cast<const uchar*>(confidence.data);
      	// for (int i = 0; i < npixels; i++)
      	// {
      	// 	  con(i) = float(pfc[i])/255.0f;
        // }
        //

        filtering_time = (float)getTickCount();

      	std::chrono::steady_clock::time_point start_solver = std::chrono::steady_clock::now();
        compute_factorization(reference, spatialSigma, lumaSigma, chromaSigma);

        // filter(target,confidence,target);
        solve(target,confidence,target);

      	std::chrono::steady_clock::time_point end_solver = std::chrono::steady_clock::now();
      	std::cout << "solver time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_solver - start_solver).count() << "ms" << std::endl;


        // Divide through by the homogeneous coordinate and store the
        // result back to the image
      	// uchar *pftar = (uchar*)(target.data);
      	// for (int i = 0; i < npixels; i++)
      	// {
      	// 	  pftar[i] = tar(i) * 255;
        // }

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
        filtering_time = ((float)getTickCount() - filtering_time)/getTickFrequency();
        cout<<"Filtering time: "<<filtering_time<<"s"<<endl;

    	cv::waitKey(0);



    }



#endif //_FILT_HPP_
