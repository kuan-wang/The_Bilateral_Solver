

#ifndef _BISTOCHASTIZE_HPP_
#define _BISTOCHASTIZE_HPP_


#include <iostream>

#include "blur.hpp"
#include "diags.hpp"
#include "splat.hpp"
#include "slice.hpp"
#include "unique.hpp"
#include "testslib.hpp"
#include "csrmatrix.hpp"
#include "hashcoords.hpp"
#include "getvalididx.hpp"
#include "factorization.hpp"


    void bistochastize(int maxiter = 10)
    {
        Eigen::VectorXf ones_npixels = Eigen::VectorXf::Ones(npixels);
        Eigen::VectorXf n = Eigen::VectorXf::Ones(nvertices);
        Eigen::VectorXf m(nvertices);
        Eigen::VectorXf bluredn(nvertices);
        Splat(ones_npixels,m);

        for (int i = 0; i < maxiter; i++) {
            Blur(n,bluredn);
            n = ((n.array()*m.array()).array()/bluredn.array()).array().sqrt();
        }

        Blur(n,bluredn);
        m = n.array() * bluredn.array();

        // diags(m,Dm);
        // diags(n,Dn);
        Dm = m.asDiagonal();
        Dn = n.asDiagonal();

    }

    void test_bistochastize()
    {
        std::vector<float> coords_flat = generateRandomVector<float>(npixels*dim);
        compute_factorization(coords_flat);

        bistochastize();

        std::cout << "Dn:" << std::endl;
        std::cout << Dn << std::endl;
        std::cout << "Dm:" << std::endl;
        std::cout << Dm << std::endl;
        std::cout << "S*S.transpose():" << std::endl;
        std::cout << S*S.transpose() << std::endl;

    }


#endif //_BISTOCHASTIZE_HPP_
