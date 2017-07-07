
#ifndef _BLUR_HPP_
#define _BLUR_HPP_






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

        result = x * (2.0 * dim);
        for (int i = 0; i < dim; i++) {
            result = result + blurs[i] * x;
        }

    }










#endif //_BLUR_HPP_
