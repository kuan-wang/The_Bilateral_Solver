
// #ifndef npixels
// #define npixels 30
// #endif

#ifndef _GETVALIDIDX_HPP_
#define _GETVALIDIDX_HPP_

#include <iostream>
#include <algorithm>
#include <vector>

#include "testslib.hpp"
#include "binarySearch.hpp"


    void get_valid_idx(std::unordered_map<long long, int>& valid, Eigen::Matrix<long long, Eigen::Dynamic, 1>& candidates,
                        Eigen::SparseMatrix<float>& blur_temp)
    {

        for (int i = 0; i < candidates.size(); i++)
        {
            if(valid.find(candidates(i)) != valid.end())
            {
                blur_temp.insert(valid[candidates(i)], i) = 1.0;
            }
        }
    }

    void get_valid_idx(std::unordered_map<float, int>& valid, Eigen::VectorXf& candidates,
                        Eigen::SparseMatrix<float>& blur_temp)
    {

        for (int i = 0; i < candidates.size(); i++)
        {
            if(valid.find(candidates(i)) != valid.end())
            {
                blur_temp.insert(valid[candidates(i)], i) = 1.0;
            }
        }
    }


    void get_valid_idx(std::vector<long long>& valid, Eigen::Matrix<long long, Eigen::Dynamic, 1>& candidates,
                        Eigen::SparseMatrix<float>& blur_temp)
    {

        for (int i = 0; i < candidates.size(); i++)
        {

            int id = binarySearchRecursive<long long>(&valid[0],0,valid.size()-1,candidates(i));   //size()-1?
            if(id >= 0)
            {
                triple_blur.push_back(Eigen::Triplet<float>(id, i, 1.0));
                // blur_temp.insert(id, i) = 1;
            }
        }

    }


    void get_valid_idx(std::vector<float>& valid, Eigen::VectorXf& candidates,
                        Eigen::SparseMatrix<float>& blur_temp)
    {

        for (int i = 0; i < candidates.size(); i++)
        {

            int id = binarySearchRecursive<float>(&valid[0],0,valid.size()-1,candidates(i));   //size()-1?
            if(id >= 0)
            {
                // std::cout << "(i,id) : (" << i <<","<< id <<")"<< std::endl;
                blur_temp.insert(id, i) = 1;
                // blur_temp.insert(i, id) = 1;
                // std::cout << "(i,id) : (" << i <<","<< id <<")"<< std::endl;
            }
        }
    }


    void get_valid_idx(std::vector<float>& valid, std::vector<float>& candidates,
                        std::vector<int>& valid_idx, std::vector<int>& locs)
    {
        valid_idx.clear();
        locs.clear();
        for (int i = 0; i < candidates.size(); i++)
        {
            int id = binarySearchRecursive<float>(&valid[0],0,valid.size(),candidates[i]);
            if(id >= 0)
            {
                locs.push_back(id);
                valid_idx.push_back(i);
            }
        }
        // std::cout << "candidates.size()" << candidates.size() << std::endl;
        // std::cout << "valid_idx.size():"<< valid_idx.size() << std::endl;

    }

    void test_get_valid_idx()
    {
        std::cout << "/ntest_get_valid_idx/n" << std::endl;


        std::vector<float> coords_flat = generateRandomVector<float>(npixels);
        std::sort(coords_flat.begin(),coords_flat.end());
        std::vector<float> neighbor_flat = generateRandomVector<float>(npixels);
        // std::sort(neighbor_flat.begin(), neighbor_flat.end());

        Eigen::SparseMatrix<float> blur_temp(npixels,npixels);
        Eigen::VectorXf unique_hashes(npixels);
        Eigen::VectorXf neighbor_hashes(npixels);

        for (int i = 0; i < npixels; i++) {
            unique_hashes(i) = coords_flat[i];
        }

        for (int k = 0; k < npixels; k++) {
            neighbor_hashes(k) = neighbor_flat[k];
        }

        std::cout << "unique_hashes" << std::endl;
        PrintVector(coords_flat);
        std::cout << "neighbor_hashes" << std::endl;
        std::cout << neighbor_hashes << std::endl;
        get_valid_idx(coords_flat, neighbor_hashes, blur_temp);
        std::cout << "blur_temp" << std::endl;
        std::cout << blur_temp << std::endl;

        // std::vector<float> valid = generateRandomVector<float>(npixels);
        // std::vector<float> candidates = generateRandomVector<float>(npixels);
        // std::vector<int> valid_idx;
        // std::vector<int> locs;
        //
        // std::sort(valid.begin(), valid.end());
        //
        // get_valid_idx(valid, candidates, valid_idx, locs);
        //
        // std::cout << "valid:" << std::endl;
        // PrintVector(valid);
        // std::cout << "candidates:" << std::endl;
        // PrintVector(candidates);
        // // PrintUnordered_set(unique_hashes);
        // std::cout << "valid_idx" << std::endl;
        // PrintVector(valid_idx);
        // std::cout << "locs" << std::endl;
        // PrintVector(locs);


    }





#endif //_GETVALIDIDX_HPP_
