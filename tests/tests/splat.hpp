
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
        // std::cout << "x.size:" << x.size() << std::endl;
        Eigen::SparseVector<double> vres(nvertices);
        result.resize(nvertices);
        // std::cout << "nvertices:" << nvertices << std::endl;
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

    void Splat(Eigen::VectorXd& x, Eigen::VectorXd& result) {
        result = S*x;
    }

    void Splat(Eigen::MatrixXd& x, Eigen::VectorXd& result) {
        result = S*x;
    }

    void Splat(std::vector<double>& x, Eigen::SparseVector<double>& result) {
        Eigen::SparseVector<double> v(x.size());
        for (int i = 0; i < x.size(); i++) {
            v.coeffRef(i) = x[i];
        }
        result = S*v;
    }

    void Splat(Eigen::SparseVector<double>& x, Eigen::SparseVector<double>& result) {
        result = S*x;
    }

    void test_splat() {

        clock_t now;

        std::vector<double> coords_flat = generateRandomVector<double>(npixels*dim);
        compute_factorization(coords_flat);

        std::vector<double> testx = generateRandomVector<double>(npixels);
        std::vector<double> result;
        Eigen::VectorXd VectorXd_result;
        Eigen::SparseVector<double> SparseVector_result;
        now = clock();
        printf( "compute_factorization : now is %f seconds\n\n", (double)(now) / CLOCKS_PER_SEC);

        Eigen::VectorXd testv(testx.size());
        for (int i = 0; i < testx.size(); i++) {
            testv(i) = testx[i];
        }
        now = clock();
        printf( "testv : now is %f seconds\n\n", (double)(now) / CLOCKS_PER_SEC);

        Eigen::SparseVector<double> spv(testx.size());
        for (int i = 0; i < testx.size(); i++) {
            spv.coeffRef(i) = testx[i];
        }
        now = clock();
        printf( "spv : now is %f seconds\n\n", (double)(now) / CLOCKS_PER_SEC);

        Splat(testx, SparseVector_result);
        now = clock();
        printf( "Splat V -> spV : now is %f seconds\n\n", (double)(now) / CLOCKS_PER_SEC);
        // std::cout << SparseVector_result << std::endl;
        Splat(testx, VectorXd_result);
        now = clock();
        printf( "Splat V -> Vxd : now is %f seconds\n\n", (double)(now) / CLOCKS_PER_SEC);
        // std::cout << VectorXd_result << std::endl;
        Splat(testx, result);
        now = clock();
        printf( "Splat V -> V : now is %f seconds\n\n", (double)(now) / CLOCKS_PER_SEC);
        // PrintVector(result);
        Splat(testv,VectorXd_result);
        now = clock();
        printf( "Splat Vxd -> Vxd : now is %f seconds\n\n", (double)(now) / CLOCKS_PER_SEC);

        Splat(spv,SparseVector_result);
        now = clock();
        printf( "Splat spV -> spV : now is %f seconds\n\n", (double)(now) / CLOCKS_PER_SEC);


    }


#endif //_SPLAT_HPP_
