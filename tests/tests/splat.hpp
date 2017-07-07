
#ifndef _SPLAT_HPP_
#define _SPLAT_HPP_

















void Splat(std::vector<double>& x, std::vector<double>& result) {
    Eigen::SparseVector<double> v(x.size());
    for (int i = 0; i < x.size(); i++) {
        v.coeffRef(i) = x[i];
    }
    std::cout << "x.size:" << x.size() << std::endl;
    Eigen::SparseVector<double> vres(nvertices);
    std::cout << "nvertices:" << nvertices << std::endl;
    std::cout << "S.size" << S.transpose().rows() <<" x "<<S.transpose().cols() << std::endl;
    vres = S.transpose()*v;
    // std::cout << "vres:" <<  (S.transpose()*v).rows() << std::endl;
    // std::cout << "vres:" <<  (S.transpose()*v).cols() << std::endl;
    //
    // std::cout << "testm*testv" << std::endl;
    // Eigen::SparseVector<double> testv(4);
    // Eigen::SparseMatrix<double> testm(4,5);
    // for (size_t i = 0; i < 4; i++) {
    //     for (size_t j = 0; j < 5; j++) {
    //         testm.insert(i,j) = (double)(i*j);
    //     }
    // }
    //
    // for (size_t i = 0; i < 4; i++) {
    //     testv.coeffRef(i) = (double)i;
    // }
    // std::cout << "testm" << testm << std::endl;
    // std::cout << "testv" << testv << std::endl;
    // std::cout << "testm*testv" << testm.transpose()*testv << std::endl;

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


#endif //_SPLAT_HPP_
