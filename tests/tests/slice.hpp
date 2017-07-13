
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



    void Slice(std::vector<double>& y, std::vector<double>& result) {
        Eigen::SparseVector<double> v(y.size());
        for (int i = 0; i < y.size(); i++) {
            v.coeffRef(i) = y[i];
        }
        Eigen::SparseVector<double> vres(nvertices);
        vres = S.transpose()*v;

        for(Eigen::SparseVector<double>::InnerIterator it(vres); it; ++it)
        {
            result[it.index()] = it.value();
        }
    }


    void Slice(Eigen::VectorXd& y, std::vector<double>& result) {
        Eigen::VectorXd vres(nvertices);
        vres = S.transpose()*y;

        for(int i = 0; i < vres.size();i++)
        {
            result[i] = vres(i);
        }
    }

    void Slice(Eigen::SparseVector<double>& y, std::vector<double>& result) {
        Eigen::SparseVector<double> vres(nvertices);
        vres = S.transpose()*y;

        for(int i = 0; i < vres.size();i++)
        {
            result[i] = vres.coeff(i);
        }
    }

    void Slice(Eigen::VectorXd& y,
               Eigen::MatrixXd& result)
    {
        result = S.transpose()*y;
    }





    void test_slice()
    {
        std::vector<double> coords_flat = generateRandomVector<double>(npixels*dim);
        compute_factorization(coords_flat);

        std::vector<double> onesx(npixels,1);
        std::vector<double> testv(npixels,-1);
        std::vector<double> onesv(npixels,-1);
        std::vector<double> testx = generateRandomVector<double>(npixels);
        std::vector<double> result;
        Eigen::VectorXd VectorXd_result;
        Eigen::SparseVector<double> SparseVector_result;
        Eigen::SparseVector<double> SparseVector_onesresult;

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
