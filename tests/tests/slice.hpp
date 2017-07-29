
#ifndef _SLICE_HPP_
#define _SLICE_HPP_


#include <iostream>

#include "unique.hpp"
#include "splat.hpp"
#include "testslib.hpp"
#include "csrmatrix.hpp"
#include "hashcoords.hpp"
#include "getvalididx.hpp"
#include "factorization.hpp"



    void Slice(std::vector<float>& y, std::vector<float>& result) {
        Eigen::SparseVector<float> v(y.size());
        for (int i = 0; i < y.size(); i++) {
            v.coeffRef(i) = y[i];
        }
        Eigen::SparseVector<float> vres(nvertices);
        vres = S.transpose()*v;

        for(Eigen::SparseVector<float>::InnerIterator it(vres); it; ++it)
        {
            result[it.index()] = it.value();
        }
    }


    void Slice(Eigen::VectorXf& y, std::vector<float>& result) {
        Eigen::VectorXf vres(nvertices);
        vres = S.transpose()*y;

        for(int i = 0; i < vres.size();i++)
        {
            result[i] = vres(i);
        }
    }

    void Slice(Eigen::SparseVector<float>& y, std::vector<float>& result) {
        Eigen::SparseVector<float> vres(nvertices);
        vres = S.transpose()*y;

        for(int i = 0; i < vres.size();i++)
        {
            result[i] = vres.coeff(i);
        }
    }

    void Slice(Eigen::VectorXf& y,
               Eigen::MatrixXf& result)
    {
        result = S.transpose()*y;
    }





    void test_slice()
    {
        std::vector<float> coords_flat = generateRandomVector<float>(npixels*dim);
        compute_factorization(coords_flat);

        std::vector<float> onesx(npixels,1);
        std::vector<float> testv(npixels,-1);
        std::vector<float> onesv(npixels,-1);
        std::vector<float> testx = generateRandomVector<float>(npixels);
        std::vector<float> result;
        Eigen::VectorXf VectorXd_result;
        Eigen::SparseVector<float> SparseVector_result;
        Eigen::SparseVector<float> SparseVector_onesresult;

        Splat(testx, VectorXd_result);
        Splat(onesx, SparseVector_onesresult);
        // Splat(testx, VectorXd_result);
        // Splat(testx, result);
        PrintVector(testx);
        std::cout << VectorXd_result << std::endl;

        Slice(VectorXd_result, testv);
        Slice(SparseVector_onesresult, onesv);
        PrintVector(testv);
        PrintVector(onesv);
        // Slice(testx, VectorXd_result);
        // std::cout << VectorXd_result << std::endl;
        // Slice(testx, result);
        // PrintVector(result);
        // std::cout << S << std::endl;


    }



#endif //_SLICE_HPP_
