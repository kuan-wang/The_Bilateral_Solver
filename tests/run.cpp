


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

// #define _DEBUG_TEST_
#ifdef _DEBUG_TEST_
using namespace std;
using namespace Eigen;

    int rows=1000000, cols = 1000000;
    std::vector<int> v1 = generateRandomVector<int>(rows);
    std::vector<int> v2 = generateRandomVector<int>(cols);
    Eigen::SparseMatrix<float, Eigen::ColMajor> mat1(rows,cols);         // 默认列优先
    Eigen::SparseMatrix<float, Eigen::RowMajor> mat2(cols,rows);         // 默认列优先
    Eigen::SparseMatrix<float> mat3(rows,rows);         // 默认列优先
    Eigen::SparseMatrix<float> mat4(cols,cols);         // 默认列优先
    // mat.reserve(VectorXi::Constant(cols,1)); //关键：为每一列保留6个非零元素空间

	  std::chrono::steady_clock::time_point start_insert1 = std::chrono::steady_clock::now();
    for (int i = 0; i < rows; i++) {
        mat1.insert(v1[i],i) = 1.0f;
    }
	  std::chrono::steady_clock::time_point end_insert1 = std::chrono::steady_clock::now();
    for (int i = 0; i < cols; i++) {
        mat2.insert(i,v2[i]) = 1.0f;
    }
	  std::chrono::steady_clock::time_point end_insert2 = std::chrono::steady_clock::now();
    for (int i = 0; i < 10; i++) {
        mat3 = mat1*mat2;
    }
	  std::chrono::steady_clock::time_point end_mul1 = std::chrono::steady_clock::now();
    for (int i = 0; i < 10; i++) {
        mat4 = mat2*mat1;
    }
	  std::chrono::steady_clock::time_point end_mul2 = std::chrono::steady_clock::now();
	  std::cout << "insert1: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_insert1 - start_insert1).count() << "ms" << std::endl;
	  std::cout << "insert2: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_insert2 - end_insert1).count() << "ms" << std::endl;
	  std::cout << "mul1: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_mul1 - end_insert2).count() << "ms" << std::endl;
	  std::cout << "mul2: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_mul2 - end_mul1).count() << "ms" << std::endl;

#endif
    // for(int i=0; i<7; i++){ //遍历行
        // for(int j=0;j<7; j++){
            // int v_ij = i+j+1;
            // mat.insert(i,j) = v_ij;                    // alternative: mat.coeffRef(i,j) += v_ij;
        // }
    // }
//     // mat.makeCompressed(); //压缩剩余的空间
//     mat1 = mat.middleRows(0,3);
//     cout << mat << endl;
//     cout << mat1 << endl;

    //
    // int n = 1000000;
    // Eigen::SparseMatrix<float> test(n,n);
    // std::vector<Eigen::Triplet<float> > triples;
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
    //     // triples.push_back(Eigen::Triplet<float>(it->second, i, float(i)));
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
    // //     // test.insert(v1[i],v2[i]) = float(i);
    // //     std::vector<int> v3(100,1);
    // // }
    // // // std::vector<Eigen::Triplet<float> > triples(n);
    // // for (int i = 0; i < n; i++) {
    //     // triples[i] = (Eigen::Triplet<float>(i, i, float(i)));
    //     // triples.push_back(Eigen::Triplet<float>(i, i, float(i)));
    // // }
	  // // std::chrono::steady_clock::time_point end_triples = std::chrono::steady_clock::now();
	  // // std::cout << "end triples: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_triples - start_test).count() << "ms" << std::endl;
    // // // test.setFromTriplets(triples.begin(), triples.end());
    // // // std::cout << "triples.size:"<<triples.size() << '\n';
    // //
    // // for (int i = 0; i < n; i++) {
    // //     test.coeffRef(i,i) = float(i);
    // //     // test.insert(v1[i],v2[i]) = float(i);
    // // }
    //
	  // std::chrono::steady_clock::time_point end_test = std::chrono::steady_clock::now();
	  // // std::cout << "test insert: " << std::chrono::duration_cast<std::chrono::milliseconds>(start_find - start_insert).count() << "ms" << std::endl;
	  // std::cout << "test construct: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_test - start_insert).count() << "ms" << std::endl;



    return 0;
}
