
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


    void get_valid_idx(std::vector<double>& valid, std::vector<double>& candidates,
                        std::vector<int>& valid_idx, std::vector<int>& locs)
    {
        valid_idx.clear();
        locs.clear();
        for (int i = 0; i < candidates.size(); i++) {
            int id = binarySearchRecursive(&valid[0],0,candidates.size(),candidates[i]);
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


        std::vector<double> valid = generateRandomVector<double>(npixels);
        std::vector<double> candidates = generateRandomVector<double>(npixels);
        std::vector<int> valid_idx;
        std::vector<int> locs;

        std::sort(valid.begin(), valid.end());

        get_valid_idx(valid, candidates, valid_idx, locs);

        std::cout << "valid:" << std::endl;
        PrintVector(valid);
        std::cout << "candidates:" << std::endl;
        PrintVector(candidates);
        // PrintUnordered_set(unique_hashes);
        std::cout << "valid_idx" << std::endl;
        PrintVector(valid_idx);
        std::cout << "locs" << std::endl;
        PrintVector(locs);


    }





#endif //_GETVALIDIDX_HPP_
