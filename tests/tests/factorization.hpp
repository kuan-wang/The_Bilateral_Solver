

// #ifndef npixels
// #define npixels 10
// #endif
//
// #ifndef dim
// #define dim 5
// #endif



#ifndef _FACTORIZATION_HPP_
#define _FACTORIZATION_HPP_

#include <set>
#include <vector>

#include "hashcoords.hpp"
#include "unique.hpp"
#include "csrmatrix.hpp"
#include "getvalididx.hpp"
// #include "bistochastize.hpp"
#include "testslib.hpp"


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
        // Eigen::SparseMatrix<float, Eigen::RowMajor>(
        Eigen::SparseMatrix<float> S_temp(npixels,npixels);
        // S_temp.reserve(Eigen::VectorXi::Constant(npixels,1));
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
              S_temp.insert(vert_idx,pix_idx) = 1.0f;
      				++vert_idx;
      			}
      			else
      			{
      				// tripletList.push_back(T(it->second, pix_idx, 1.0f));
              S_temp.insert(it->second,pix_idx) = 1.0f;
      			}

      			pref += 3; // skip 3 bytes (y u v)
      			++pix_idx;
      		}
      	}
        nvertices = hashed_coords.size();
        // S_temp.makeCompressed();
        S = Eigen::SparseMatrix<float>(nvertices, npixels);
        S = S_temp.middleRows(0, nvertices);
        S.finalize();
        std::cout << "nvertices:"<<nvertices << '\n';
        // std::cout << S_temp.nonZeros() << '\n';
        // std::cout << S_temp.rows()<<"x"<<S_temp.cols() << '\n';
        // std::cout << S.nonZeros() << '\n';
        // std::cout << S.rows()<<"x"<<S.cols() << '\n';
      	std::chrono::steady_clock::time_point end_splat_construction = std::chrono::steady_clock::now();
      	std::cout << "S construction: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_splat_construction - begin_grid_construction).count() << "ms" << std::endl;



      	// Blur matrices   // Eigen::RowMajor or Eigen::ColMajor???
      	std::chrono::steady_clock::time_point begin_blur_construction = std::chrono::steady_clock::now();
        // blurs_test = Eigen::SparseMatrix<float>(nvertices,nvertices);
        Eigen::VectorXf ones_nvertices = Eigen::VectorXf::Ones(nvertices);
        Eigen::VectorXf ones_npixels = Eigen::VectorXf::Ones(npixels);
        blurs_test = ones_nvertices.asDiagonal();
        blurs_test *= 10;
        for(int offset = -1; offset <= 1;++offset)
        {
            if(offset == 0) continue;
          	for (int i = 0; i < 5; ++i)
          	{
          	     Eigen::SparseMatrix<float> blur_temp(hashed_coords.size(), hashed_coords.size());
                //  blur_temp.reserve(Eigen::VectorXi::Constant(nvertices,6));
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


    void compute_factorization(Eigen::Matrix<long long, Eigen::Dynamic, Eigen::Dynamic>& coords_flat)
    {
        Eigen::Matrix<long long, Eigen::Dynamic, 1> hashed_coords;
        Eigen::Matrix<long long, Eigen::Dynamic, Eigen::Dynamic> unique_coords;
        std::vector<long long> unique_hashes;
        // std::unordered_map<float,int> unique_hashes;

        clock_t now;
        now = clock();
        printf( "start hashcoords : now is %f seconds\n\n", (float)(now) / CLOCKS_PER_SEC);
        hash_coords(coords_flat,hashed_coords);
        // std::cout << "coords_flat:" << std::endl;
        // std::cout << coords_flat << std::endl;
        // std::cout << "hashed_coords:" << std::endl;
        // std::cout << hashed_coords << std::endl;

        now = clock();
        printf( "start unique : now is %f seconds\n\n", (float)(now) / CLOCKS_PER_SEC);
        unique(coords_flat, unique_coords, hashed_coords, unique_hashes);
        S.finalize();
        std::cout << "finish unique()" << std::endl;

        // std::cout << "unique_coords:" << std::endl;
        // std::cout << unique_coords << std::endl;
        // std::cout << "unique_hashes:" << std::endl;
        // std::set<float>::iterator iter=unique_hashes.begin();
        // while(iter!=unique_hashes.end())
        // {
        //     std::cout<<*iter<<std::endl;
        //     ++iter;
        // }


        now = clock();
        printf( "start construct blur : now is %f seconds\n\n", (float)(now) / CLOCKS_PER_SEC);
        blurs_test = Eigen::SparseMatrix<float>(nvertices,nvertices);
        Eigen::VectorXf bl = Eigen::VectorXf::Ones(nvertices);
        Eigen::Matrix<long long, Eigen::Dynamic, 1> onesx = Eigen::Matrix<long long, Eigen::Dynamic, 1>::Ones(nvertices);
        Eigen::SparseMatrix<float> blur_temp(nvertices,nvertices);
        blur_temp = bl.asDiagonal()*2*dim;
        for (int i = 0; i < dim; i++) {
            Eigen::SparseMatrix<float> blur(nvertices,nvertices);
            for (int j = -1; j <= 1; j++) {
                if(j == 0) continue;
                // Eigen::SparseMatrix<float> blur_temp(nvertices,nvertices);
                Eigen::Matrix<long long, Eigen::Dynamic, Eigen::Dynamic> neighbor_coords = unique_coords;
                Eigen::Matrix<long long, Eigen::Dynamic, 1> neighbor_hashes;

                neighbor_coords.col(i) = neighbor_coords.col(i) + j*onesx;

                hash_coords(neighbor_coords,neighbor_hashes);
                get_valid_idx(unique_hashes, neighbor_hashes, blur_temp);
                // blurs_test = blurs_test + blur_temp;

            }
            // std::cout << "blur:"<< blur << std::endl;
            std::cout << "blur"<< i << std::endl;
            // blurs.push_back(blur);
        }
        blurs_test.setFromTriplets(triple_blur.begin(), triple_blur.end());
        blurs_test = blurs_test + blur_temp;
        blurs_test.finalize();
        now = clock();
        printf( "finished construct blur.finalize() : now is %f seconds\n\n", (float)(now) / CLOCKS_PER_SEC);

        // std::cout << "S:" << std::endl;
        // std::cout << S.cols()<<S.rows() << std::endl;
        // std::cout << "unique_hashes:" << std::endl;
        // PrintVector(unique_hashes);


    }


    void compute_factorization(Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& coords_flat)
    {
        Eigen::Matrix<long long, Eigen::Dynamic, 1> hashed_coords;
        Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> unique_coords;
        std::vector<long long> unique_hashes;
        // std::unordered_map<float,int> unique_hashes;

        clock_t now;
        now = clock();
        printf( "start hashcoords : now is %f seconds\n\n", (float)(now) / CLOCKS_PER_SEC);
        hash_coords(coords_flat,hashed_coords);
        // std::cout << "coords_flat:" << std::endl;
        // std::cout << coords_flat << std::endl;
        // std::cout << "hashed_coords:" << std::endl;
        // std::cout << hashed_coords << std::endl;

        now = clock();
        printf( "start unique : now is %f seconds\n\n", (float)(now) / CLOCKS_PER_SEC);
        unique(coords_flat, unique_coords, hashed_coords, unique_hashes);
        std::cout << "finish unique()" << std::endl;

        // std::cout << "unique_coords:" << std::endl;
        // std::cout << unique_coords << std::endl;
        // std::cout << "unique_hashes:" << std::endl;
        // std::set<float>::iterator iter=unique_hashes.begin();
        // while(iter!=unique_hashes.end())
        // {
        //     std::cout<<*iter<<std::endl;
        //     ++iter;
        // }


        now = clock();
        printf( "start construct blur : now is %f seconds\n\n", (float)(now) / CLOCKS_PER_SEC);
        blurs_test = Eigen::SparseMatrix<float>(nvertices,nvertices);
        Eigen::VectorXf bl = Eigen::VectorXf::Ones(nvertices);
        Eigen::Matrix<unsigned char, Eigen::Dynamic, 1> onesx = Eigen::Matrix<unsigned char, Eigen::Dynamic, 1>::Ones(nvertices);
        blurs_test = bl.asDiagonal()*2*dim;
        for (int i = 0; i < dim; i++) {
            Eigen::SparseMatrix<float> blur(nvertices,nvertices);
            for (int j = -1; j <= 1; j++) {
                if(j == 0) continue;
                Eigen::SparseMatrix<float> blur_temp(nvertices,nvertices);
                Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> neighbor_coords = unique_coords;
                Eigen::Matrix<long long, Eigen::Dynamic, 1> neighbor_hashes;

                // std::vector<int> valid_coord;
                // std::vector<int> neighbor_idx;
                // std::vector<int> valid_idx;
                // for (int k = 0; k < nvertices; k++) {
                //     neighbor_coords(k,i) += j;
                // }
                neighbor_coords.col(i) = neighbor_coords.col(i) + j*onesx;

                hash_coords(neighbor_coords,neighbor_hashes);
            // std::cout << "neighbor_coords:" << std::endl;
            // std::cout << neighbor_coords << std::endl;
                get_valid_idx(unique_hashes, neighbor_hashes, blur_temp);
                // get_valid_idx(unique_hashes,neighbor_hashes,valid_coord,neighbor_idx);
                // std::cout <<i<<j<< "nvertices,valid_coord.size,neighbor_idx.size:"<< nvertices<<valid_coord.size()<<neighbor_idx.size() << std::endl;
            // std::cout << "ones_valid_coord:" << std::endl;
            // PrintVector(ones_valid_coord);
            // std::cout << "valid_coord:" << std::endl;
            // PrintVector(valid_coord);
            // std::cout << "neighbor_idx:" << std::endl;
            // PrintVector(neighbor_idx);
            //     std::cout << "blur_temp:"<< blur_temp << std::endl;
                // blur = blur + blur_temp;
                blurs_test = blurs_test + blur_temp;
                // std::cout << blur << std::endl;
            }
            // std::cout << "blur:"<< blur << std::endl;
            std::cout << "blur"<< i << std::endl;
            // blurs.push_back(blur);
        }
        now = clock();
        printf( "finished construct blur : now is %f seconds\n\n", (float)(now) / CLOCKS_PER_SEC);

        // std::cout << "S:" << std::endl;
        // std::cout << S.cols()<<S.rows() << std::endl;
        // std::cout << "unique_hashes:" << std::endl;
        // PrintVector(unique_hashes);


    }


    void compute_factorization(Eigen::MatrixXf& coords_flat)
    {
        Eigen::VectorXf hashed_coords;
        Eigen::MatrixXf unique_coords;
        std::vector<float> unique_hashes;
        // std::unordered_map<float,int> unique_hashes;

        clock_t now;
        now = clock();
        printf( "start hashcoords : now is %f seconds\n\n", (float)(now) / CLOCKS_PER_SEC);
        hash_coords(coords_flat,hashed_coords);
        // std::cout << "coords_flat:" << std::endl;
        // std::cout << coords_flat << std::endl;
        // std::cout << "hashed_coords:" << std::endl;
        // std::cout << hashed_coords << std::endl;

        now = clock();
        printf( "start unique : now is %f seconds\n\n", (float)(now) / CLOCKS_PER_SEC);
        unique(coords_flat, unique_coords, hashed_coords, unique_hashes);
        std::cout << "finish unique()" << std::endl;

        // std::cout << "unique_coords:" << std::endl;
        // std::cout << unique_coords << std::endl;
        // std::cout << "unique_hashes:" << std::endl;
        // std::set<float>::iterator iter=unique_hashes.begin();
        // while(iter!=unique_hashes.end())
        // {
        //     std::cout<<*iter<<std::endl;
        //     ++iter;
        // }


        now = clock();
        printf( "start construct blur : now is %f seconds\n\n", (float)(now) / CLOCKS_PER_SEC);
        blurs_test = Eigen::SparseMatrix<float>(nvertices,nvertices);
        Eigen::VectorXf bl = Eigen::VectorXf::Ones(nvertices);
        blurs_test = bl.asDiagonal()*2*dim;
        for (int i = 0; i < dim; i++) {
            Eigen::SparseMatrix<float> blur(nvertices,nvertices);
            for (int j = -1; j <= 1; j++) {
                if(j == 0) continue;
                Eigen::SparseMatrix<float> blur_temp(nvertices,nvertices);
                Eigen::MatrixXf neighbor_coords = unique_coords;
                Eigen::VectorXf neighbor_hashes;

                // std::vector<int> valid_coord;
                // std::vector<int> neighbor_idx;
                // std::vector<int> valid_idx;
                // for (int k = 0; k < nvertices; k++) {
                //     neighbor_coords(k,i) += j;
                // }
                neighbor_coords.col(i) = neighbor_coords.col(i) + j*bl;

                hash_coords(neighbor_coords,neighbor_hashes);
            // std::cout << "neighbor_coords:" << std::endl;
            // std::cout << neighbor_coords << std::endl;
                get_valid_idx(unique_hashes, neighbor_hashes, blur_temp);
                // get_valid_idx(unique_hashes,neighbor_hashes,valid_coord,neighbor_idx);
                // std::cout <<i<<j<< "nvertices,valid_coord.size,neighbor_idx.size:"<< nvertices<<valid_coord.size()<<neighbor_idx.size() << std::endl;
            // std::cout << "ones_valid_coord:" << std::endl;
            // PrintVector(ones_valid_coord);
            // std::cout << "valid_coord:" << std::endl;
            // PrintVector(valid_coord);
            // std::cout << "neighbor_idx:" << std::endl;
            // PrintVector(neighbor_idx);
            //     std::cout << "blur_temp:"<< blur_temp << std::endl;
                // blur = blur + blur_temp;
                blurs_test = blurs_test + blur_temp;
                // std::cout << blur << std::endl;
            }
            // std::cout << "blur:"<< blur << std::endl;
            std::cout << "blur"<< i << std::endl;
            // blurs.push_back(blur);
        }
        now = clock();
        printf( "finished construct blur : now is %f seconds\n\n", (float)(now) / CLOCKS_PER_SEC);

        // std::cout << "S:" << std::endl;
        // std::cout << S.cols()<<S.rows() << std::endl;
        // std::cout << "unique_hashes:" << std::endl;
        // PrintVector(unique_hashes);


    }


    void compute_factorization(std::vector<float>& coords_flat)
    {
        std::vector<float> hashed_coords;
        std::vector<float> unique_hashes;
        std::vector<float> unique_coords;
        std::vector<int> unique_idx;
        std::vector<int> idx;
        // std::vector<float> ones_npixels(npixels,1.0);
        std::vector<int> arange_npixels(npixels);
        // std::vector<int> test_npixels(npixels*30);

        clock_t now;
        now = clock();
        printf( "start test for : now is %f seconds\n\n", (float)(now) / CLOCKS_PER_SEC);
    // #pragma omp parallel for num_threads(4)
    // #pragma omp parallel for
        for (int i = 0; i < npixels; i++) {
            arange_npixels[i] = i;
        }
    // // #pragma omp parallel for num_threads(4)
    // // #pragma omp parallel for
    //     for (int i = 0; i < npixels*10; i++) {
    //         test_npixels[i] = i;
    //     }
    // // #pragma omp parallel for num_threads(4)
    // // #pragma omp parallel for
    //     for (int i = 0; i < npixels*10; i++) {
    //         test_npixels[i] = i;
    //     }
    // // #pragma omp parallel for num_threads(4)
    // // #pragma omp parallel for
    //     for (int i = 0; i < npixels*10; i++) {
    //         test_npixels[i] = i;
    //     }

        now = clock();
        printf( "end for : now is %f seconds\n\n", (float)(now) / CLOCKS_PER_SEC);

        std::cout << "start hash_coords(coords_flat,hash_coords)" << std::endl;
        hash_coords(coords_flat,hashed_coords);

        unique(hashed_coords,unique_hashes,unique_idx,idx);
        std::cout << "finish unique()" << std::endl;

        // std::cout << "hashed_coords:" << std::endl;
        // PrintVector(hashed_coords);
        // std::cout << "unique_hashes:" << std::endl;
        // PrintVector(unique_hashes);
        // std::cout << "unique_idx:" << std::endl;
        // PrintVector(unique_idx);
        // std::cout << "idx:" << std::endl;
        // PrintVector(idx);


        nvertices = unique_idx.size();
        S = Eigen::SparseMatrix<float>(nvertices,npixels);

        std::cout << "start Construct csr_matrix S" << std::endl;
        // csr_matrix(S, ones_npixels, idx, arange_npixels);
        csr_matrix(S, idx, arange_npixels);


        unique_coords.resize(nvertices*dim);
        for (int i = 0; i < nvertices; i++) {
            for (int j = 0; j < dim; j++) {
                unique_coords[i*dim+j] = (coords_flat[unique_idx[i]*dim+j]);
            }
        }

        std::cout << "start Construct blurs" << std::endl;
        for (int i = 0; i < dim; i++) {
            Eigen::SparseMatrix<float> blur(nvertices,nvertices);
            for (int j = -1; j <= 1; j++) {
                if(j == 0) continue;
                Eigen::SparseMatrix<float> blur_temp(nvertices,nvertices);
                std::vector<float> neighbor_coords = unique_coords;
                std::vector<float> neighbor_hashes;
                std::vector<int> valid_coord;
                std::vector<int> neighbor_idx;
                std::vector<int> valid_idx;
                for (int k = 0; k < nvertices; k++) {
                    neighbor_coords[k*dim+i] += j;
                }
                hash_coords(neighbor_coords,neighbor_hashes);
            // std::cout << "neighbor_coords:" << std::endl;
            // PrintVector(neighbor_coords);
            // std::cout << "neighbor_hashes:" << std::endl;
            // PrintVector(neighbor_hashes);
                get_valid_idx(unique_hashes,neighbor_hashes,valid_coord,neighbor_idx);
                // std::vector<float> ones_valid_coord(valid_coord.size(),1.0);
                std::cout <<i<<j<< "nvertices,valid_coord.size,neighbor_idx.size:"<< nvertices<<" "<<valid_coord.size()<<" "<<neighbor_idx.size() << std::endl;
                csr_matrix(blur_temp, valid_coord, neighbor_idx);
            // std::cout << "ones_valid_coord:" << std::endl;
            // PrintVector(ones_valid_coord);
            // std::cout << "valid_coord:" << std::endl;
            // PrintVector(valid_coord);
            // std::cout << "neighbor_idx:" << std::endl;
            // PrintVector(neighbor_idx);
                // std::cout << "blur_temp:"<< blur_temp << std::endl;
                blur = blur + blur_temp;
            }
            // std::cout << "blur:"<< blur << std::endl;
            std::cout << "blur"<< i << std::endl;
            blurs.push_back(blur);
        }


    }

    void test_compute_factorization()
    {
        // std::vector<float> coords_flat = generateRandomVector<float>(npixels*dim);
        // Eigen::MatrixXf coords_flat = Eigen::MatrixXf::Random(npixels,5);
        // coords_flat = 100.0*(coords_flat + Eigen::MatrixXf::Ones(npixels,5));
        Eigen::MatrixXf coords_flat(npixels*dim,1);
        std::vector<float> randV = generateRandomVector<float>(npixels*dim);
        for (int i = 0; i < randV.size(); i++) {
            coords_flat(i,0) = randV[i];
        }
        coords_flat.resize(npixels,dim);
        // std::cout << "coords_flat" << std::endl;
        // std::cout << coords_flat << std::endl;

        compute_factorization(coords_flat);

        // std::cout << "S:" << std::endl;
        // std::cout << S << std::endl;
        // std::cout << "blurs:" << std::endl;
        // PrintVector(blurs);

    }


#endif //_FACTORIZATION_HPP_
