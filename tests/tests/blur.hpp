
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


    void Blur(Eigen::VectorXf& x, Eigen::VectorXf& result)
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



    void Blur(std::vector<float>& x, std::vector<float>& result)
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
            Eigen::SparseVector<float> v(x.size());
            for (int j = 0; j < x.size(); j++)
            {
                v.coeffRef(j) = x[j];
            }
            Eigen::SparseVector<float> blured(nvertices);
            blured = blurs[i] * v;

            for(Eigen::SparseVector<float>::InnerIterator it(blured); it; ++it)
            {
                result[it.index()] += it.value();
            }
        }
    }

    void Blur(Eigen::SparseMatrix<float>& x, Eigen::SparseMatrix<float>& result)
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
        std::vector<float> coords_flat = generateRandomVector<float>(npixels*dim);
        compute_factorization(coords_flat);

        std::vector<float> onesx(npixels,1);
        std::vector<float> testv(npixels,-1);
        std::vector<float> onesv(npixels,-1);
        std::vector<float> testx = generateRandomVector<float>(npixels);
        std::vector<float> spalt_result;
        std::vector<float> blur_result;
        std::vector<float> slice_result(npixels, -1);

        std::vector<float> spalt_onesresult;
        std::vector<float> blur_onesresult;
        std::vector<float> slice_onesresult(npixels, -1);

        Eigen::VectorXf VectorXd_result;
        Eigen::SparseVector<float> SparseVector_result;
        Eigen::SparseVector<float> SparseVector_onesresult;

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
