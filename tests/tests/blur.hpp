
#ifndef _BLUR_HPP_
#define _BLUR_HPP_

#include "splat.hpp"
#include "slice.hpp"
#include "unique.hpp"
#include "testslib.hpp"
#include "csrmatrix.hpp"
#include "hashcoords.hpp"
#include "getvalididx.hpp"
#include "factorization.hpp"


    void Blur(Eigen::VectorXd& x, Eigen::VectorXd& result)
    {
        if(x.size() != nvertices)
        {
            std::cout << "x.size() != nvertices" << std::endl;
            exit(-1);
        }

        // result = (2*dim)*x;
        // for (int i = 0; i < dim; i++)
        // {
        //     result = result + blurs[i] * x;
        // }

        result = blurs_test * x;

    }



    void Blur(std::vector<double>& x, std::vector<double>& result)
    {
        if(x.size() != nvertices)
        {
            std::cout << "x.size() != nvertices" << std::endl;
            exit(-1);
        }
        result.resize(nvertices);

        for (int i = 0; i < nvertices; i++) {
            result[i] = 2 * dim * x[i];
        }
        for (int i = 0; i < dim; i++) {
            Eigen::SparseVector<double> v(x.size());
            for (int j = 0; j < x.size(); j++)
            {
                v.coeffRef(j) = x[j];
            }
            Eigen::SparseVector<double> blured(nvertices);
            blured = blurs[i] * v;

            for(Eigen::SparseVector<double>::InnerIterator it(blured); it; ++it)
            {
                result[it.index()] += it.value();
            }
        }
    }

    void Blur(Eigen::SparseMatrix<double>& x, Eigen::SparseMatrix<double>& result)
    {
        if(x.rows() != nvertices || x.cols() != nvertices)
        {
            std::cout << "x.rows() != nvertices || x.cols() != nvertices" << std::endl;
            exit(-1);
        }

        // result = x * (2.0 * dim);
        // for (int i = 0; i < dim; i++) {
        //     result = result + blurs[i] * x;
        // }
        result = blurs_test * x;

    }


    void test_blur()
    {
        std::vector<double> coords_flat = generateRandomVector<double>(npixels*dim);
        compute_factorization(coords_flat);

        std::vector<double> onesx(npixels,1);
        std::vector<double> testv(npixels,-1);
        std::vector<double> onesv(npixels,-1);
        std::vector<double> testx = generateRandomVector<double>(npixels);
        std::vector<double> spalt_result;
        std::vector<double> blur_result;
        std::vector<double> slice_result(npixels, -1);

        std::vector<double> spalt_onesresult;
        std::vector<double> blur_onesresult;
        std::vector<double> slice_onesresult(npixels, -1);

        Eigen::VectorXd VectorXd_result;
        Eigen::SparseVector<double> SparseVector_result;
        Eigen::SparseVector<double> SparseVector_onesresult;

        Splat(testx, spalt_result);
        Splat(onesx, spalt_onesresult);
        // Splat(onesx, SparseVector_onesresult);
        // Splat(testx, VectorXd_result);
        // Splat(testx, result);
        std::cout << "testx:" << std::endl;
        PrintVector(testx);
        std::cout << "spalt_result:" << std::endl;
        PrintVector(spalt_result);
        Blur(spalt_result, blur_result);
        Blur(spalt_onesresult, blur_onesresult);
        std::cout << "blur_result:" << std::endl;
        PrintVector(blur_result);
        Slice(blur_result, slice_result);
        Slice(blur_onesresult, slice_onesresult);
        std::cout << "slice_result:" << std::endl;
        PrintVector(slice_result);
        std::cout << "slice_onesresult:" << std::endl;
        PrintVector(slice_onesresult);


        // Slice(VectorXd_result, testv);
        // Slice(SparseVector_onesresult, onesv);
        // PrintVector(testv);
        // PrintVector(onesv);

        // Slice(testx, VectorXd_result);
        // std::cout << VectorXd_result << std::endl;
        // Slice(testx, result);
        // PrintVector(result);
        // std::cout << S << std::endl;


    }









#endif //_BLUR_HPP_
