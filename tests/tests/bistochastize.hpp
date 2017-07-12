

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


    void bistochastize(int maxiter = 1)
    {
        std::vector<double> ones_npixels(npixels,1.0);
        std::vector<double> n(nvertices,1.0);
        std::vector<double> m(nvertices);
        Splat(ones_npixels,m);

        std::cout << "m:" << std::endl;
        for (int i = 0; i < maxiter; i++) {
            std::vector<double> bluredn;
            Blur(n,bluredn);
            for (int j = 0; j < n.size(); j++) {
                n[j] = sqrtf(n[j]*m[j]/bluredn[j]);
            }
        }

        std::vector<double> bluredn;
        Blur(n,bluredn);
        for (int i = 0; i < n.size(); i++) {
            m[i] = n[i]*bluredn[i];
        }

        diags(m,Dm);
        diags(n,Dn);

    }

    void test_bistochastize()
    {
        std::vector<double> coords_flat = generateRandomVector<double>(npixels*dim);
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
