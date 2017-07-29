

#ifndef _CSRMATRIX_HPP_
#define _CSRMATRIX_HPP_

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>


#include <iostream>
#include <vector>

#include "testslib.hpp"


    void csr_matrix(Eigen::SparseMatrix<float>& spmat, std::vector<float>& values,
                    std::vector<int>& rows, std::vector<int>& cols)
    {
        for (int i = 0; i < values.size(); i++) {
            spmat.insert(rows[i],cols[i]) = values[i];
        }
    }

    void csr_matrix(Eigen::SparseMatrix<float>& spmat, std::vector<int>& rows, std::vector<int>& cols)
    {
        for (int i = 0; i < rows.size(); i++) {
            spmat.insert(rows[i],cols[i]) = 1;
        }
    }


    void test_csr_matrix()
    {
        Eigen::SparseMatrix<float> spmat(npixels*2, npixels*3);
        std::vector<float> values = generateRandomVector<float>(npixels*2);
        std::vector<int> rows = generateRandomVector<int>(npixels*2);
        std::vector<int> cols = generateRandomVector<int>(npixels*2);

        csr_matrix(spmat, values, rows, cols);

        std::cout << "values:" << std::endl;
        PrintVector(values);
        std::cout << "rows:" << std::endl;
        PrintVector(rows);
        std::cout << "cols:" << std::endl;
        PrintVector(cols);
        std::cout << "spmat:" << std::endl;
        std::cout << spmat << std::endl;


    }



#endif //_CSRMATRIX_HPP_
