
#ifndef _DIAGS_HPP_
#define _DIAGS_HPP_

#include <iostream>

#include "testslib.hpp"

    void diags(Eigen::VectorXf& v,Eigen::SparseMatrix<float>& m)
    {
        m = Eigen::SparseMatrix<float>(v.size(),v.size());
        for (int i = 0; i < v.size(); i++) {
            m.insert(i,i) = v(i);
        }
    }

    void diags(std::vector<float>& v,Eigen::SparseMatrix<float>& m)
    {
        m = Eigen::SparseMatrix<float>(v.size(),v.size());
        for (int i = 0; i < v.size(); i++) {
            m.insert(i,i) = v[i];
        }
    }

    void test_diags()
    {
        std::vector<float> v = generateRandomVector<float>(npixels);
        Eigen::SparseMatrix<float> mat;
        diags(v, mat);
        std::cout << "v:" << std::endl;
        PrintVector(v);
        std::cout << "mat:" << std::endl;
        std::cout << mat << std::endl;
    }



#endif //_DIAGS_HPP_
