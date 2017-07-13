

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
#include "testslib.hpp"



    void compute_factorization(Eigen::MatrixXd& coords_flat)
    {
        Eigen::VectorXd hashed_coords;
        Eigen::MatrixXd unique_coords;
        std::vector<double> unique_hashes;

        clock_t now;
        now = clock();
        printf( "start hashcoords : now is %f seconds\n\n", (double)(now) / CLOCKS_PER_SEC);
        hash_coords(coords_flat,hashed_coords);
        // std::cout << "coords_flat:" << std::endl;
        // std::cout << coords_flat << std::endl;
        // std::cout << "hashed_coords:" << std::endl;
        // std::cout << hashed_coords << std::endl;

        now = clock();
        printf( "start unique : now is %f seconds\n\n", (double)(now) / CLOCKS_PER_SEC);
        unique(coords_flat, unique_coords, hashed_coords, unique_hashes);
        std::cout << "finish unique()" << std::endl;

        // std::cout << "unique_coords:" << std::endl;
        // std::cout << unique_coords << std::endl;
        // std::cout << "unique_hashes:" << std::endl;
        // std::set<double>::iterator iter=unique_hashes.begin();
        // while(iter!=unique_hashes.end())
        // {
        //     std::cout<<*iter<<std::endl;
        //     ++iter;
        // }


        now = clock();
        printf( "start construct blur : now is %f seconds\n\n", (double)(now) / CLOCKS_PER_SEC);
        for (int i = 0; i < dim; i++) {
            Eigen::SparseMatrix<double> blur(nvertices,nvertices);
            for (int j = -1; j <= 1; j++) {
                if(j == 0) continue;
                Eigen::SparseMatrix<double> blur_temp(nvertices,nvertices);
                Eigen::MatrixXd neighbor_coords = unique_coords;
                Eigen::VectorXd neighbor_hashes;

                // std::vector<int> valid_coord;
                // std::vector<int> neighbor_idx;
                // std::vector<int> valid_idx;
                for (int k = 0; k < nvertices; k++) {
                    neighbor_coords(k,i) += j;
                }
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
                blur = blur + blur_temp;
                // std::cout << blur << std::endl;
            }
            // std::cout << "blur:"<< blur << std::endl;
            std::cout << "blur"<< i << std::endl;
            blurs.push_back(blur);
        }


    }


    void compute_factorization(std::vector<double>& coords_flat)
    {
        std::vector<double> hashed_coords;
        std::vector<double> unique_hashes;
        std::vector<double> unique_coords;
        std::vector<int> unique_idx;
        std::vector<int> idx;
        // std::vector<double> ones_npixels(npixels,1.0);
        std::vector<int> arange_npixels(npixels);
        // std::vector<int> test_npixels(npixels*30);

        clock_t now;
        now = clock();
        printf( "start test for : now is %f seconds\n\n", (double)(now) / CLOCKS_PER_SEC);
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
        printf( "end for : now is %f seconds\n\n", (double)(now) / CLOCKS_PER_SEC);

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
        S = Eigen::SparseMatrix<double>(nvertices,npixels);

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
            Eigen::SparseMatrix<double> blur(nvertices,nvertices);
            for (int j = -1; j <= 1; j++) {
                if(j == 0) continue;
                Eigen::SparseMatrix<double> blur_temp(nvertices,nvertices);
                std::vector<double> neighbor_coords = unique_coords;
                std::vector<double> neighbor_hashes;
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
                // std::vector<double> ones_valid_coord(valid_coord.size(),1.0);
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
        // std::vector<double> coords_flat = generateRandomVector<double>(npixels*dim);
        // Eigen::MatrixXd coords_flat = Eigen::MatrixXd::Random(npixels,5);
        // coords_flat = 100.0*(coords_flat + Eigen::MatrixXd::Ones(npixels,5));
        Eigen::MatrixXd coords_flat(npixels*dim,1);
        std::vector<double> randV = generateRandomVector<double>(npixels*dim);
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
