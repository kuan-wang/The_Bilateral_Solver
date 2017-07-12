
#ifndef _SPLAT_HPP_
#define _SPLAT_HPP_


#include <iostream>

#include "unique.hpp"
#include "testslib.hpp"
#include "csrmatrix.hpp"
#include "hashcoords.hpp"
#include "getvalididx.hpp"
#include "factorization.hpp"




    void Splat(std::vector<double>& x, std::vector<double>& result) {
        Eigen::SparseVector<double> v(x.size());
        for (int i = 0; i < x.size(); i++) {
            v.coeffRef(i) = x[i];
        }
        std::cout << "x.size:" << x.size() << std::endl;
        Eigen::SparseVector<double> vres(nvertices);
        result.resize(nvertices);
        std::cout << "nvertices:" << nvertices << std::endl;
        vres = S * v;

        for(Eigen::SparseVector<double>::InnerIterator it(vres); it; ++it)
        {
            result[it.index()] = it.value();
        }
    }

    void Splat(std::vector<double>& x, Eigen::VectorXd& result) {
        Eigen::VectorXd v(x.size());
        for (int i = 0; i < x.size(); i++) {
            v(i) = x[i];
        }
        result = S*v;
    }

    void Splat(std::vector<double>& x, Eigen::SparseVector<double>& result) {
        Eigen::SparseVector<double> v(x.size());
        for (int i = 0; i < x.size(); i++) {
            v.coeffRef(i) = x[i];
        }
        result = S*v;
    }

    void test_splat() {
        std::vector<double> coords_flat = generateRandomVector<double>(npixels*dim);
        compute_factorization(coords_flat);

        std::vector<double> testx = generateRandomVector<double>(npixels);
        std::vector<double> result;
        Eigen::VectorXd VectorXd_result;
        Eigen::SparseVector<double> SparseVector_result;

        Splat(testx, SparseVector_result);
        std::cout << SparseVector_result << std::endl;
        // Splat(testx, VectorXd_result);
        // std::cout << VectorXd_result << std::endl;
        // Splat(testx, result);
        // PrintVector(result);
        PrintVector(testx);
        // std::cout << S << std::endl;


    }


#endif //_SPLAT_HPP_
