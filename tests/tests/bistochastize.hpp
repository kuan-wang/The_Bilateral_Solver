

#ifndef _BISTOCHASTIZE_HPP_
#define _BISTOCHASTIZE_HPP_




    void bistochastize(int maxiter = 10) {
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


#endif //_BISTOCHASTIZE_HPP_
