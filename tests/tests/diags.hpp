
#ifndef _DIAGS_HPP_
#define _DIAGS_HPP_



    void diags(std::vector<double>& v,Eigen::SparseMatrix<double>& m) {
        m = Eigen::SparseMatrix<double>(v.size(),v.size());
        for (int i = 0; i < v.size(); i++) {
            m.insert(i,i) = v[i];
        }
    }


#endif //_DIAGS_HPP_
