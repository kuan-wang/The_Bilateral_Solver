

#ifndef _UNIQUE_HPP_
#define _UNIQUE_HPP_

#include <iostream>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <set>


#include "testslib.hpp"
#include "binarySearch.hpp"



    void unique(Eigen::Matrix<long long, Eigen::Dynamic, Eigen::Dynamic>& coords_flat,
                Eigen::Matrix<long long, Eigen::Dynamic, Eigen::Dynamic>& unique_coords,
                Eigen::Matrix<long long, Eigen::Dynamic, 1>& hashed_coords,
                std::unordered_map<long long,int>& unique_hashes)
    {
        unique_hashes.clear();
        clock_t now;
        now = clock();
        printf( "start input unordered_map : now is %f seconds\n\n", (double)(now) / CLOCKS_PER_SEC);
        int id = 0;
        std::vector<Eigen::Triplet<double> > triple;
        std::vector<int> idx;
        for (int i = 0; i < hashed_coords.size(); i++) {
            // input.insert(hashed_coords(i));
            if(unique_hashes.find(hashed_coords(i)) == unique_hashes.end())
            {
                unique_hashes[hashed_coords(i)] = id;
                triple.push_back(Eigen::Triplet<double>(id, i, 1.0));
                idx.push_back(i);
                id++;
            }
            else
            {
                triple.push_back(Eigen::Triplet<double>(unique_hashes[hashed_coords(i)], i, 1.0));
            }
        }
        nvertices = unique_hashes.size();


        now = clock();
        printf( "start resize unique_coords : now is %f seconds\n\n", (double)(now) / CLOCKS_PER_SEC);
        unique_coords.resize(unique_hashes.size(),coords_flat.cols());
        for (int i = 0; i < idx.size(); i++) {
            unique_coords.row(i) = coords_flat.row(idx[i]);
        }


        now = clock();
        printf( "start construct S : now is %f seconds\n\n", (double)(now) / CLOCKS_PER_SEC);
        S = Eigen::SparseMatrix<double>(nvertices,npixels);
        S.setFromTriplets(triple.begin(), triple.end());
        now = clock();
        printf( "end construct S : now is %f seconds\n\n", (double)(now) / CLOCKS_PER_SEC);

    }



    void unique(Eigen::MatrixXd& coords_flat, Eigen::MatrixXd& unique_coords,
                Eigen::VectorXd& hashed_coords, std::vector<long long>& unique_hashes)
    {
        unique_hashes.clear();
        std::set<double> input;
        clock_t now;
        now = clock();
        printf( "start input set : now is %f seconds\n\n", (double)(now) / CLOCKS_PER_SEC);
        for (int i = 0; i < hashed_coords.size(); i++) {
            input.insert(hashed_coords(i));
        }

        now = clock();
        printf( "start resize unique_coords : now is %f seconds\n\n", (double)(now) / CLOCKS_PER_SEC);
        unique_coords.resize(input.size(),coords_flat.cols());
        nvertices = input.size();
        unique_hashes.resize(input.size());
        S = Eigen::SparseMatrix<double>(nvertices,npixels);

        now = clock();
        printf( "start std::copy : now is %f seconds\n\n", (double)(now) / CLOCKS_PER_SEC);
        std::copy(input.begin(),input.end(),unique_hashes.begin());

        now = clock();
        printf( "start construct S : now is %f seconds\n\n", (double)(now) / CLOCKS_PER_SEC);
        for (int i = 0; i < hashed_coords.size(); i++) {
            int id = binarySearchRecursive(&unique_hashes[0],0,unique_hashes.size()-1,hashed_coords(i));  //size()-1?
            // std::set<double>::iterator got = unique_hashes.find (hashed_coords(i));
            // if(got != unique_hashes.end())
            if(id >= 0)
            {
                S.insert(id, i) = 1.0;
                unique_coords.row(id) = coords_flat.row(i);
                // std::cout << "(id,i) : (" << id <<","<< i <<")"<< std::endl;
            }
        }
        now = clock();
        printf( "end construct S : now is %f seconds\n\n", (double)(now) / CLOCKS_PER_SEC);
    }



    void unique(Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& coords_flat,
                Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& unique_coords,
                Eigen::Matrix<long long, Eigen::Dynamic, 1>& hashed_coords,
                std::vector<long long>& unique_hashes)
    {
        unique_hashes.clear();
        std::set<long long> input;
        clock_t now;
        now = clock();
        printf( "start input set : now is %f seconds\n\n", (double)(now) / CLOCKS_PER_SEC);
        for (int i = 0; i < hashed_coords.size(); i++) {
            input.insert(hashed_coords(i));
        }

        now = clock();
        printf( "start resize unique_coords : now is %f seconds\n\n", (double)(now) / CLOCKS_PER_SEC);
        unique_coords.resize(input.size(),coords_flat.cols());
        nvertices = input.size();
        unique_hashes.resize(input.size());
        S = Eigen::SparseMatrix<double>(nvertices,npixels);

        now = clock();
        printf( "start std::copy : now is %f seconds\n\n", (double)(now) / CLOCKS_PER_SEC);
        std::copy(input.begin(),input.end(),unique_hashes.begin());

        now = clock();
        printf( "start construct S : now is %f seconds\n\n", (double)(now) / CLOCKS_PER_SEC);
        for (int i = 0; i < hashed_coords.size(); i++) {
            int id = binarySearchRecursive(&unique_hashes[0],0,unique_hashes.size()-1,hashed_coords(i));  //size()-1?
            // std::set<double>::iterator got = unique_hashes.find (hashed_coords(i));
            // if(got != unique_hashes.end())
            if(id >= 0)
            {
                S.insert(id, i) = 1.0;
                unique_coords.row(id) = coords_flat.row(i);
                // std::cout << "(id,i) : (" << id <<","<< i <<")"<< std::endl;
            }
        }
        now = clock();
        printf( "end construct S : now is %f seconds\n\n", (double)(now) / CLOCKS_PER_SEC);
    }

    void unique(Eigen::MatrixXd& coords_flat, Eigen::MatrixXd& unique_coords,
                Eigen::VectorXd& hashed_coords, std::vector<double>& unique_hashes)
    {
        unique_hashes.clear();
        std::set<double> input;
        clock_t now;
        now = clock();
        printf( "start input set : now is %f seconds\n\n", (double)(now) / CLOCKS_PER_SEC);
        for (int i = 0; i < hashed_coords.size(); i++) {
            input.insert(hashed_coords(i));
        }

        now = clock();
        printf( "start resize unique_coords : now is %f seconds\n\n", (double)(now) / CLOCKS_PER_SEC);
        unique_coords.resize(input.size(),coords_flat.cols());
        nvertices = input.size();
        unique_hashes.resize(input.size());
        S = Eigen::SparseMatrix<double>(nvertices,npixels);

        now = clock();
        printf( "start std::copy : now is %f seconds\n\n", (double)(now) / CLOCKS_PER_SEC);
        std::copy(input.begin(),input.end(),unique_hashes.begin());

        now = clock();
        printf( "start construct S : now is %f seconds\n\n", (double)(now) / CLOCKS_PER_SEC);
        for (int i = 0; i < hashed_coords.size(); i++) {
            int id = binarySearchRecursive(&unique_hashes[0],0,unique_hashes.size()-1,hashed_coords(i));  //size()-1?
            // std::set<double>::iterator got = unique_hashes.find (hashed_coords(i));
            // if(got != unique_hashes.end())
            if(id >= 0)
            {
                S.insert(id, i) = 1.0;
                unique_coords.row(id) = coords_flat.row(i);
                // std::cout << "(id,i) : (" << id <<","<< i <<")"<< std::endl;
            }
        }
        now = clock();
        printf( "end construct S : now is %f seconds\n\n", (double)(now) / CLOCKS_PER_SEC);
    }


    void unique(std::vector<double>& hashed_coords, std::unordered_set<double>& unique_hashes,
                std::vector<int>& unique_idx, std::vector<int>& idx)
    {
        unique_idx.clear();
        idx.clear();
        std::cout << "for 1" << std::endl;
        std::cout << "hashed_coords size" <<hashed_coords.size()<< std::endl;
        for (int i = 0; i < hashed_coords.size(); i++) {
            unique_hashes.insert(hashed_coords[i]);
        }
        unique_idx.resize(unique_hashes.size(),-1);
        // idx.resize(npixels);
        std::cout << "unique_hashes size" <<unique_hashes.size()<< std::endl;

        std::cout << "for 2" << std::endl;
        int count = 0;
        for (int i = 0; i < hashed_coords.size(); i++) {
            // int id = binarySearchRecursive(&unique_hashes[0],0,input.size(),hashed_coords[i]);
            std::unordered_set<double>::iterator got = unique_hashes.find (hashed_coords[i]);
            if(got != unique_hashes.end())
            {
                int id = std::distance(unique_hashes.begin(), got);
                idx[count++] = id;
                if(unique_idx[id] < 0) unique_idx[id] = i;
            }
        }

        std::cout << "for 2 end" << std::endl;

    }

    void unique(std::vector<double>& hashed_coords, std::vector<double>& unique_hashes,
                std::vector<int>& unique_idx,std::vector<int>& idx)
    {

        unique_idx.clear();
        idx.clear();
        unique_hashes.clear();


        std::set<double> input;
        std::cout << "for 1" << std::endl;
        std::cout << "hashed_coords size" <<hashed_coords.size()<< std::endl;
        for (int i = 0; i < hashed_coords.size(); i++) {
            // std::cout << "hashed_coords:"<<hashed_coords[i] << std::endl;
            input.insert(hashed_coords[i]);
        }
        unique_hashes.resize(input.size());
        unique_idx.resize(input.size(),-1);
        idx.resize(hashed_coords.size());
        std::copy(input.begin(),input.end(),unique_hashes.begin());
        // std::cout << "input :" <<unique_hashes<< std::endl;
        std::cout << "input size" <<input.size()<< std::endl;

        std::cout << "for 2" << std::endl;
        clock_t now;
        now = clock();
        printf( "start test for 2 : now is %f seconds\n\n", (double)(now) / CLOCKS_PER_SEC);
        int count = 0;
        for (int i = 0; i < hashed_coords.size(); i++) {
            int id = binarySearchRecursive(&unique_hashes[0],0,input.size(),hashed_coords[i]);
            if(id >= 0)
            {
                // idx.push_back(id);
                idx[count++] = id;
                if(unique_idx[id] < 0) unique_idx[id] = i;
            }
        }
        now = clock();
        printf( "end test for 2 : now is %f seconds\n\n", (double)(now) / CLOCKS_PER_SEC);
        std::cout << "for 2 end" << std::endl;

    }


    // void unique(std::vector<double>& hashed_coords, std::vector<double>& unique_hashes,
    //             std::vector<double>& coords)
    // {
    //
    //     unique_idx.clear();
    //     idx.clear();
    //     unique_hashes.clear();
    //
    //
    //     std::set<double> input;
    //     std::cout << "for 1" << std::endl;
    //     std::cout << "hashed_coords size" <<hashed_coords.size()<< std::endl;
    //     for (int i = 0; i < hashed_coords.size(); i++) {
    //         // std::cout << "hashed_coords:"<<hashed_coords[i] << std::endl;
    //         input.insert(hashed_coords[i]);
    //     }
    //     unique_hashes.resize(input.size());
    //     unique_idx.resize(input.size(),-1);
    //     std::copy(input.begin(),input.end(),unique_hashes.begin());
    //     // std::cout << "input :" <<unique_hashes<< std::endl;
    //     std::cout << "input size" <<input.size()<< std::endl;
    //
    //     std::cout << "for 2" << std::endl;
    //     for (int i = 0; i < hashed_coords.size(); i++) {
    //         int id = binarySearchRecursive(&unique_hashes[0],0,input.size(),hashed_coords[i]);
    //         if(id >= 0)
    //         {
    //             idx.push_back(id);
    //             if(unique_idx[id] < 0) unique_idx[id] = i;
    //         }
    //     }
    //
    //     std::cout << "for 2 end" << std::endl;
    //
    // }


    void test_unique() {

        std::cout << "/ntest_unique/n" << std::endl;


        std::vector<double> hashed_coords = generateRandomVector<double>(npixels);
        // std::vector<double> hashed_coords(npixels,1.0);
        std::vector<double> unique_hashes;
        // std::unordered_set<double> unique_hashes;
        std::vector<int> unique_idx;
        std::vector<int> idx;
        unique(hashed_coords, unique_hashes, unique_idx, idx);

        std::cout << "hashed_coords:" << std::endl;
        PrintVector(hashed_coords);
        std::cout << "unique_hashes" << std::endl;
        PrintVector(unique_hashes);
        // PrintUnordered_set(unique_hashes);
        std::cout << "unique_idx" << std::endl;
        PrintVector(unique_idx);
        std::cout << "idx" << std::endl;
        PrintVector(idx);



    }






#endif //_UNIQUE_HPP_
