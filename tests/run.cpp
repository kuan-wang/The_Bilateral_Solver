


// #include "tests/unique.hpp"
// #include "tests/hashcoords.hpp"
// #include "tests/csrmatrix.hpp"
// #include "tests/diags.hpp"
// #include "tests/getvalididx.hpp"
// #include "tests/factorization.hpp"
// #include "tests/splat.hpp"
// #include "tests/slice.hpp"
// #include "tests/blur.hpp"
// #include "tests/filt.hpp"
// #include "tests/solve_color.hpp"
#include "tests/solve_comp_int8.hpp"
// #include "tests/solve_comp.hpp"
// #include "tests/solve.hpp"
// #include "tests/bistochastize.hpp"



int main(int argc, char const *argv[]) {

   args = argv;
    // test_unique();
    // test_hash_coords();
    // test_csr_matrix();
    // test_diags();
    // test_get_valid_idx();
    // test_compute_factorization();
    // test_splat();
    // test_slice();
    // test_blur();
   test_solve();
    // test_filt();
    // test_bistochastize();

    std::cout << "test" << '\n';

// using namespace std;
// using namespace Eigen;
//
//     int rows=10, cols = 10;
//     Eigen::SparseMatrix<double> mat(rows,cols);         // 默认列优先
//     Eigen::SparseMatrix<double> mat1(3,cols);         // 默认列优先
//     // mat.reserve(VectorXi::Constant(cols,1)); //关键：为每一列保留6个非零元素空间
//     for(int i=0; i<7; i++){ //遍历行
//         for(int j=0;j<7; j++){
//             int v_ij = i+j+1;
//             mat.insert(i,j) = v_ij;                    // alternative: mat.coeffRef(i,j) += v_ij;
//         }
//     }
//     // mat.makeCompressed(); //压缩剩余的空间
//     mat1 = mat.middleRows(0,3);
//     cout << mat << endl;
//     cout << mat1 << endl;

    //
    // int n = 1000000;
    // Eigen::SparseMatrix<double> test(n,n);
    // std::vector<Eigen::Triplet<double> > triples;
    // std::unordered_map<int /* hash */, int /* vert id */> hashed_coords;
	  // hashed_coords.reserve(n);
    // std::vector<int> v1(n);
    // std::set<int> set1;
    // std::vector<int> v2 = generateRandomVector<int>(n);
    // std::vector<int> v3 = generateRandomVector<int>(n);
    //
	  // std::chrono::steady_clock::time_point start_insert = std::chrono::steady_clock::now();
    // int vert_idx = 0;
    // for (int i = 0; i < n; i++) {
		//     auto it = hashed_coords.find(v2[i]);
    //     if(it == hashed_coords.end())
    //     {
    //         hashed_coords.insert(std::pair<int, int>(v2[i], vert_idx));
    //         test.coeffRef(vert_idx,i) = 1.0f;
    //         vert_idx++;
    //     }
    //     else
    //     {
    //         test.coeffRef(it->second,i) = 1.0f;
    //     }
    // }
    // // for (int i = 0; i < n; i++) {
    //   // int id = binarySearchRecursive<int>(&v1[0],0,v1.size()-1,v2[i]);  //size()-1?
    //     // triples.push_back(Eigen::Triplet<double>(it->second, i, double(i)));
    // // }
	  // std::chrono::steady_clock::time_point start_find = std::chrono::steady_clock::now();
	  // // std::cout << "insert: " << std::chrono::duration_cast<std::chrono::milliseconds>(start_find - start_insert).count() << "ms" << std::endl;
	  // std::chrono::steady_clock::time_point end_triples = std::chrono::steady_clock::now();
	  // // std::cout << "triples: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_triples - start_find).count() << "ms" << std::endl;
    // // test.setFromTriplets(triples.begin(), triples.end());
    // // std::cout << "triples.size:"<<triples.size() << '\n';
    //
    //
    // // std::vector<int> v1(n);
    // // std::vector<int> v2(n);
    // // for (int i = 0; i < n; i++) {
    // //     v1[i] = i;
    // //     v2[i] = i;
    // //     // test.insert(v1[i],v2[i]) = double(i);
    // //     std::vector<int> v3(100,1);
    // // }
    // // // std::vector<Eigen::Triplet<double> > triples(n);
    // // for (int i = 0; i < n; i++) {
    //     // triples[i] = (Eigen::Triplet<double>(i, i, double(i)));
    //     // triples.push_back(Eigen::Triplet<double>(i, i, double(i)));
    // // }
	  // // std::chrono::steady_clock::time_point end_triples = std::chrono::steady_clock::now();
	  // // std::cout << "end triples: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_triples - start_test).count() << "ms" << std::endl;
    // // // test.setFromTriplets(triples.begin(), triples.end());
    // // // std::cout << "triples.size:"<<triples.size() << '\n';
    // //
    // // for (int i = 0; i < n; i++) {
    // //     test.coeffRef(i,i) = double(i);
    // //     // test.insert(v1[i],v2[i]) = double(i);
    // // }
    //
	  // std::chrono::steady_clock::time_point end_test = std::chrono::steady_clock::now();
	  // // std::cout << "test insert: " << std::chrono::duration_cast<std::chrono::milliseconds>(start_find - start_insert).count() << "ms" << std::endl;
	  // std::cout << "test construct: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_test - start_insert).count() << "ms" << std::endl;



    return 0;
}
