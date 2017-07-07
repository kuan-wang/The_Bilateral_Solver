
#ifndef _SLICE_HPP_
#define _SLICE_HPP_


















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



#endif //_SLICE_HPP_
