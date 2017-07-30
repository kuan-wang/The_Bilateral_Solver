
#ifndef _FILT_HPP_
#define _FILT_HPP_

#define ENABLE_DOMAIN_TRANSFORM_FILTER

#include <iostream>
#include <chrono>

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
        std::vector<int> valid_idx(npixels);
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
      				// tripletList.push_back(T(vert_idx, pix_idx, 1.0f));
              // S_temp.insert(vert_idx,pix_idx) = 1.0f;
              valid_idx[pix_idx] = vert_idx;
      				++vert_idx;
      			}
      			else
      			{
      				// tripletList.push_back(T(it->second, pix_idx, 1.0f));
              // S_temp.insert(it->second,pix_idx) = 1.0f;
              valid_idx[pix_idx] = it->second;
      			}

      			pref += 3; // skip 3 bytes (y u v)
      			++pix_idx;
      		}
      	}
        nvertices = hashed_coords.size();
        // S_temp.makeCompressed();
        S = Eigen::SparseMatrix<float, Eigen::ColMajor>(nvertices, npixels);
        // S = S_temp.middleRows(0, nvertices);
        for (int i = 0; i < npixels; i++)
        {
            S.insert(valid_idx[i],i) = 1.0f;
        }
        S.finalize();
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
        Eigen::VectorXf n = ones_nvertices;
        Eigen::VectorXf m = S*ones_npixels;
        Eigen::VectorXf bluredn;

        for (int i = 0; i < maxiter; i++) {
            bluredn = blurs_test*n;
            n = ((n.array()*m.array()).array()/bluredn.array()).array().sqrt();
        }
        m = n.array() * (blurs_test*n).array();
        // diags(m,Dm);
        // diags(n,Dn);
        Dm = m.asDiagonal();
        Dn = n.asDiagonal();
      	std::chrono::steady_clock::time_point end_bisto_construction = std::chrono::steady_clock::now();
      	std::cout << "Dm Dn construction: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_bisto_construction - end_blur_construction).count() << "ms" << std::endl;


    }


    void solve(Eigen::MatrixXf& x,
               Eigen::MatrixXf& w,
               Eigen::MatrixXf& out)
    {



      	std::chrono::steady_clock::time_point start_solve = std::chrono::steady_clock::now();

        // SparseMatrix<float> A_diag(nvertices);
        Eigen::SparseMatrix<float, Eigen::ColMajor> M(nvertices,nvertices);
        Eigen::SparseMatrix<float, Eigen::ColMajor> A_data(nvertices,nvertices);
        Eigen::SparseMatrix<float, Eigen::ColMajor> A(nvertices,nvertices);
        Eigen::VectorXf b(nvertices);
        Eigen::VectorXf y(nvertices);
        Eigen::VectorXf w_splat(nvertices);
        Eigen::VectorXf xw(x.size());

      	std::chrono::steady_clock::time_point start_A_construction = std::chrono::steady_clock::now();
      	std::cout << "before A construction: " << std::chrono::duration_cast<std::chrono::milliseconds>(start_A_construction - start_solve).count() << "ms" << std::endl;

	    // std::cout << "start Splat(w,w_splat)" << std::endl;
        // w_splat = S*w;
        A_data = (S*w).asDiagonal();
        // A.noalias() = bs_param.lam * (Dm - Dn * (blurs_test*Dn)) + A_data ;
        A = bs_param.lam * (Dm - Dn * (blurs_test*Dn)) + A_data ;
      	std::chrono::steady_clock::time_point end_A_construction = std::chrono::steady_clock::now();
      	std::cout << "A construction: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_A_construction - start_A_construction).count() << "ms" << std::endl;

        xw = x.array() * w.array();
        b = S*xw;
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

        Eigen::SparseMatrix<float, Eigen::ColMajor> Sli = (S.transpose());
        out.noalias() = Sli*y;
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
        cv::Mat filtered_disp;
        cv::ximgproc::fastGlobalSmootherFilter(reference, target, filtered_disp, fgs_spatialSigma, fgs_colorSigma);
        // cv::ximgproc::fastGlobalSmootherFilter(reference, target, filtered_disp, spatialSigma*spatialSigma, lumaSigma);

      	std::chrono::steady_clock::time_point end_fgs = std::chrono::steady_clock::now();
      	std::cout << "fastGlobalSmootherFilter time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_fgs - start_fgs).count() << "ms" << std::endl;
        cv::equalizeHist(filtered_disp, filtered_disp);
        cv::imshow("fgs_filtered_disp",filtered_disp);






    	  cvtColor(reference, reference, cv::COLOR_BGR2YUV);
    	  // cvtColor(reference, reference, cv::COLOR_BGR2YCrCb);
        std::cout << "reference:" << reference.cols<<"x"<< reference.rows<< std::endl;
        npixels = reference.cols*reference.rows;

        cv::Mat r(npixels, 5, CV_8U);
        cv::Mat t(npixels, 1, CV_32F);
        cv::Mat c(npixels, 1, CV_32F);
        // std::vector<float> re(reference.cols*reference.rows*5);
        // std::vector<float> ta(reference.cols*reference.rows);
        // std::vector<float> co(reference.cols*reference.rows);
        int idx = 0;
    	std::cout << "start filling positions and values" << std::endl;
        now = clock();
        printf( "fill positions and values : now is %f seconds\n", (float)(now) / CLOCKS_PER_SEC);
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
                float *datac = c.ptr<float>(idx);
                // datac[0] = 1.0;
                datac[0] = float(confidence.at<uchar>(y,x))/255.0;
                // if(datac[0] != 0) datac[0] = 0.1/datac[0];
                // datac[0] = confidence.at<float>(x,y)[0];
                idx++;
            }
        }
        idx = 0;
        for (int y = 0; y < reference.rows; y++) {
            for (int x = 0; x < reference.cols; x++) {
                float *datat = t.ptr<float>(idx);
                datat[0] = float(target.at<uchar>(y,x))/255.0;
                // datat[0] = target.at<cv::Vec3b>(x,y)[0];
                idx++;
            }
        }

        std::cout << "cv2eigen" << std::endl;
        Eigen::Matrix<uchar, Eigen::Dynamic, Eigen::Dynamic> ref_temp;
        Eigen::Matrix<long long, Eigen::Dynamic, Eigen::Dynamic> ref;
        Eigen::MatrixXf tar;
        Eigen::MatrixXf con;

        cv::cv2eigen(r,ref_temp);
        cv::cv2eigen(t,tar);
        cv::cv2eigen(c,con);
        ref = ref_temp.cast<long long>();
        std::cout << "finished cv2eigen" << std::endl;




        filtering_time = (float)getTickCount();

      	std::chrono::steady_clock::time_point start_solver = std::chrono::steady_clock::now();
        compute_factorization(reference, spatialSigma, lumaSigma, chromaSigma);

        solve(tar,con,tar);

      	std::chrono::steady_clock::time_point end_solver = std::chrono::steady_clock::now();
      	std::cout << "solver time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_solver - start_solver).count() << "ms" << std::endl;


        // Divide through by the homogeneous coordinate and store the
        // result back to the image
        idx = 0;
        for (int y = 0; y < reference.rows; y++) {
            for (int x = 0; x < reference.cols; x++) {
                // float w = values[idx*4+3];
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
        filtering_time = ((float)getTickCount() - filtering_time)/getTickFrequency();
        cout<<"Filtering time: "<<filtering_time<<"s"<<endl;

    	cv::waitKey(0);



    }



#endif //_FILT_HPP_
