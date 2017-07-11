

// #ifndef npixels
// #define npixels 10
// #endif
//
// #ifndef dim
// #define dim 5
// #endif



#ifndef _FACTORIZATION_HPP_
#define _FACTORIZATION_HPP_


#include "hashcoords.hpp"
#include "unique.hpp"
#include "csrmatrix.hpp"
#include "getvalididx.hpp"
#include "testslib.hpp"


Eigen::SparseMatrix<double> S;
std::vector<Eigen::SparseMatrix<double> > blurs;

    void compute_factorization(std::vector<double>& coords_flat)
    {
        std::vector<double> hashed_coords;
        std::vector<double> unique_hashes;
        std::vector<double> unique_coords;
        std::vector<int> unique_idx;
        std::vector<int> idx;
        std::vector<double> ones_npixels(npixels,1.0);
        std::vector<int> arange_npixels(npixels);

        for (int i = 0; i < npixels; i++) {
            arange_npixels.push_back(i);
        }

        std::cout << "start hash_coords(coords_flat,hash_coords)" << std::endl;
        hash_coords(coords_flat,hashed_coords);

        unique(hashed_coords,unique_hashes,unique_idx,idx);
        std::cout << "finish unique()" << std::endl;

        nvertices = unique_idx.size();
        S = Eigen::SparseMatrix<double>(npixels,nvertices);
        for (int i = 0; i < nvertices; i++) {
            for (int j = 0; j < dim; j++) {
                unique_coords.push_back(coords_flat[unique_idx[i]*dim+j]);
            }
        }

        std::cout << "start Construct csr_matrix S" << std::endl;
        csr_matrix(S, ones_npixels, idx, arange_npixels);


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
                get_valid_idx(unique_hashes,neighbor_hashes,valid_coord,neighbor_idx);
                std::vector<double> ones_valid_coord(valid_coord.size(),1.0);
                // std::cout <<i<<j<< "nvertices,valid_coord.size,neighbor_idx.size:"<< nvertices<<valid_coord.size()<<neighbor_idx.size() << std::endl;
                csr_matrix(blur_temp, ones_valid_coord, valid_coord, neighbor_idx);
                // std::cout << "blur_temp:"<< blur_temp << std::endl;
                blur = blur + blur_temp;
            }
            std::cout << "blur"<< i << std::endl;
            blurs.push_back(blur);
        }


    }

    void test_compute_factorization()
    {
        std::vector<double> coords_flat = generateRandomVector<double>(npixels*dim);
        compute_factorization(coords_flat);

        std::cout << "S:" << std::endl;
        std::cout << S << std::endl;
        std::cout << "blurs:" << std::endl;
        PrintVector(blurs);

    }


#endif //_FACTORIZATION_HPP_
