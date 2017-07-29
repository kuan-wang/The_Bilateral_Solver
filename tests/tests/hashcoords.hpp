


#ifndef _HASHCOORDS_HPP_
#define _HASHCOORDS_HPP_

// #include<opencv2/core/core.hpp>
// #include<opencv2/core/eigen.hpp>
// #include<opencv2/highgui.hpp>
// #include<opencv2/opencv.hpp>

#include "testslib.hpp"


    void hash_coords(Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& coords_flat,
                     Eigen::Matrix<long long, Eigen::Dynamic, 1>& hashed_coords)
    {
        long long max_val = 255;
        Eigen::Matrix<long long, 5, 1> hash_vec;
        hash_vec(0) = 1;
        for (int i = 1; i < 5; i++) {
            hash_vec[i] = hash_vec[i-1]*max_val;
        }
        hashed_coords = coords_flat.cast<long long>() * hash_vec;
    }

    void hash_coords(Eigen::Matrix<long long, Eigen::Dynamic, Eigen::Dynamic>& coords_flat,
                     Eigen::Matrix<long long, Eigen::Dynamic, 1>& hashed_coords)
    {
        long long max_val = 255;
        Eigen::Matrix<long long, 5, 1> hash_vec;
        hash_vec(0) = 1;
        for (int i = 1; i < 5; i++) {
            hash_vec[i] = hash_vec[i-1]*max_val;
        }
        hashed_coords = coords_flat * hash_vec;
    }

    void hash_coords(Eigen::MatrixXf& coords_flat, Eigen::VectorXf& hashed_coords)
    {
        float max_val = 255.0;
        Eigen::VectorXf hash_vec(5);
        hash_vec[0] = 1;
        for (int i = 1; i < 5; i++) {
            hash_vec[i] = hash_vec[i-1]*max_val;
        }
        hashed_coords = coords_flat * hash_vec;
    }


    void hash_coords(std::vector<float>& coords_flat, std::vector<float>& hashed_coords)
    {
        float max_val = 255.0;
        hashed_coords.clear();
        // hashed_coords.resize(coords_flat.size()/dim);
        for (int i = 0; i < coords_flat.size()/dim; i++) {
            float hash = 0;
            for (int j = 0; j < dim; j++) {
                hash = coords_flat[i*dim+j] + hash*max_val;
                // std::cout << "hash:"<< hash << std::endl;
            }
            hashed_coords.push_back(hash);
            // hashed_coords[i] = hash;
        }
        std::cout << "hashed_coords == nvertices" <<hashed_coords.size()<<nvertices << std::endl;
        std::cout << "coords_flat.size()/dim"<<coords_flat.size()/dim << std::endl;
    }



    void test_hash_coords() {
        std::vector<float> coords_flat = generateRandomVector<float>(npixels*dim);
        std::vector<float> hashed_coords = generateRandomVector<float>(npixels);

        hash_coords(coords_flat,hashed_coords);

        std::cout << "coords_flat:" << std::endl;
        PrintVector(coords_flat);
        std::cout << "hashed_coords:" << std::endl;
        PrintVector(hashed_coords);
    }


#endif //_HASHCOORDS_HPP_
