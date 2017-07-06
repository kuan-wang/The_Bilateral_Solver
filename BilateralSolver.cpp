
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>




#include<opencv2/core/core.hpp>
// #include<opencv2/core/eigen.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/opencv.hpp>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <algorithm>
#include <iostream>
#include <iterator>
#include <vector>

#include <math.h>
#include <cmath>
#include <vector>
#include <set>
#include <memory>
//
// // #include "Sparse-Matrix/src/SparseMatrix/SparseMatrix.cpp"
// // #include "ICCG.hpp"
//
//
//
// // using namespace Eigen;
// using std::vector;
//
// // Hash table implementation for permutohedral lattice.
// //
// // The lattice points are stored sparsely using a hash table.
// // The key for each point is its spatial location in the (d+1)-
// // dimensional space.
// class HashTablePermutohedral {
// public:
//     // Hash table constructor
//     // kd : the dimensionality of the position vectors
//     // vd : the dimensionality of the value vectors
//     HashTablePermutohedral(int kd, int vd) : kd(kd), vd(vd) {
//         filled = 0;
//         entries.resize(1 << 15);
//         keys.resize(kd*entries.size()/2);
//         values.resize(vd*entries.size()/2, 0.0f);
//     }
//
//     // Returns the number of vectors stored.
//     int size() { return filled; }
//
//     // Returns a pointer to the keys array.
//     vector<short> &getKeys() { return keys; }
//
//     // Returns a pointer to the values array.
//     vector<double> &getValues() { return values; }
//
//     // Looks up the value vector associated with a given key. May or
//     // may not create a new entry if that key doesn’t exist.
//     double *lookup(const vector<short> &key, bool create = true) {
//         // Double hash table size if necessary
//         if (create && filled >= entries.size()/2) { grow(); }
//
//         // Hash the key
//         size_t h = hash(&key[0]) % entries.size();
//
//         // Find the entry with the given key
//         while (1) {
//             Entry e = entries[h];
//
//             // Check if the cell is empty
//             if (e.keyIdx == -1) {
//                 if (!create) return NULL;// Not found
//
//                 // Need to create an entry. Store the given key.
//                 for (int i = 0; i < kd; i++) {
//                     keys[filled*kd+i] = key[i];
//                 }
//                 e.keyIdx = filled*kd;
//                 e.valueIdx = filled*vd;
//                 entries[h] = e;
//                 filled++;
//                 return &values[e.valueIdx];
//             }
//
//             // check if the cell has a matching key
//             bool match = true;
//
//             for (int i = 0; i < kd && match; i++) {
//                 match = keys[e.keyIdx+i] == key[i];
//             }
//             if (match) {
//                 return &values[e.valueIdx];
//             }
//
//             // increment the bucket with wraparound
//             h++;
//             if (h == entries.size()) { h = 0; }
//         }
//     }
//
//     // Hash function used in this implementation. A simple base conversion.
//     size_t hash(const short *key) {
//         size_t h = 0;
//         for (int i = 0; i < kd; i++) {
//             h += key[i];
//             h *= 2531011;
//         }
//         return h;
//     }
// private:
//     // Grows the hash table when it runs out of space
//     void grow() {
//         // Grow the arrays
//         values.resize(vd*entries.size(), 0.0f);
//         keys.resize(kd*entries.size());
//         vector<Entry> newEntries(entries.size()*2);
//
//         // Rehash all the entries
//         for (size_t i = 0; i < entries.size(); i++) {
//             if (entries[i].keyIdx == -1) { continue; }
//             size_t h = hash(&keys[entries[i].keyIdx]) % newEntries.size();
//             while (newEntries[h].keyIdx != -1) {
//                 h++;
//                 if (h == newEntries.size()) { h = 0; }
//             }
//             newEntries[h] = entries[i];
//         }
//         entries.swap(newEntries);
//     }
//
//     // Private struct for the hash table entries.
//     struct Entry {
//         Entry() : keyIdx(-1), valueIdx(-1) {}
//         int keyIdx;
//         int valueIdx;
//     };
//
//     vector<short> keys;
//     vector<double> values;
//     vector<Entry> entries;
//     size_t filled;
//     int kd, vd;
// };
// // The algorithm class that performs the filter
// //
// // PermutohedralLattice::filter(...) does all the work.
// //
// class PermutohedralLattice {
// public:
//     // Performs a Gauss transform
//     // pos : position vectors
//     // pd : position dimensions
//     // val : value vectors
//     // vd : value dimensions
//     // n : number of items to filter
//     // out : place to store the output
//     static void filter(const double *pos, int pd,
//                         const double *val, int vd,
//                         int n, double *out) {
//
//         clock_t now;
//
//         // Create lattice
//         PermutohedralLattice lattice(pd, vd, n);
//
// 	    std::cout << "start splat" << std::endl;
//         now = clock();
//         printf( "now is %f seconds\n", (double)(now) / CLOCKS_PER_SEC);
//         // Splat
//         for (int i = 0; i < n; i++) {
//             lattice.splat(pos + i*pd, val + i*vd);
//         }
// 	    std::cout << "end splat" << std::endl;
//         now = clock();
//         printf( "now is %f seconds\n", (double)(now) / CLOCKS_PER_SEC);
//
//
// 	    std::cout << "start blur" << std::endl;
//         now = clock();
//         printf( "now is %f seconds\n", (double)(now) / CLOCKS_PER_SEC);
//         // Blur
//         lattice.blur();
// 	    std::cout << "end blur" << std::endl;
//         now = clock();
//         printf( "now is %f seconds\n", (double)(now) / CLOCKS_PER_SEC);
//
//
// 	    std::cout << "start slice" << std::endl;
//         now = clock();
//         printf( "now is %f seconds\n", (double)(now) / CLOCKS_PER_SEC);
//         // Slice
//         lattice.beginSlice();
//         for (int i = 0; i < n; i++) {
//             lattice.slice(out + i*vd);
//         }
// 	    std::cout << "end slice" << std::endl;
//         now = clock();
//         printf( "now is %f seconds\n", (double)(now) / CLOCKS_PER_SEC);
//     }
//
//     // Permutohedral lattice constructor
//     // pd : dimensionality of position vectors
//     // vd : dimensionality of value vectors
//     // n : number of points in the input
//     PermutohedralLattice(int pd, int vd, int n) :
//         d(pd), vd(vd), n(n), hashTable(pd, vd) {
//
//         // Allocate storage for various arrays
//         elevated.resize(d+1);
//         scaleFactor.resize(d);
//         greedy.resize(d+1);
//         rank.resize(d+1);
//         barycentric.resize(d+2);
//         canonical.resize((d+1)*(d+1));
//         key.resize(d+1);
//         replay.resize(n*(d+1));
//         nReplay = 0;
//
//         // compute the coordinates of the canonical simplex, in which
//         // the difference between a contained point and the zero
//         // remainder vertex is always in ascending order.
//         for (int i = 0; i <= d; i++) {
//             for (int j = 0; j <= d-i; j++) {
//                 canonical[i*(d+1)+j] = i;
//             }
//             for (int j = d-i+1; j <= d; j++) {
//                 canonical[i*(d+1)+j] = i - (d+1);
//             }
//         }
//
//         // Compute part of the rotation matrix E that elevates a
//         // position vector into the hyperplane
//         for (int i = 0; i < d; i++) {
//             // the diagonal entries for normalization
//             scaleFactor[i] = 1.0f/(sqrtf((double)(i+1)*(i+2)));
//
//             // We presume that the user would like to do a Gaussian
//             // blur of standard deviation 1 in each dimension (or a
//             // total variance of d, summed over dimensions.) Because
//             // the total variance of the blur performed by this
//             // algorithm is not d, we must scale the space to offset
//             // this.
//             //
//             // The total variance of the algorithm is:
//             // [variance of splatting] +
//             // [variance of blurring] +
//             // [variance of splatting]
//             // = d(d+1)(d+1)/12 + d(d+1)(d+1)/2 + d(d+1)(d+1)/12
//             // = 2d(d+1)(d+1)/3.
//             //
//             // So we need to scale the space by (d+1)sqrt(2/3).
//
//             scaleFactor[i] *= (d+1)*sqrtf(2.0/3);
//         }
//     }
//
//     // Performs splatting with given position and value vectors
//     void splat(const double *position, const double *value) {
//
//         // First elevate position into the (d+1)-dimensional hyperplane
//         elevated[d] = -d*position[d-1]*scaleFactor[d-1];
//         for (int i = d-1; i > 0; i--)
//         elevated[i] = (elevated[i+1] -
//             i*position[i-1]*scaleFactor[i-1] +
//             (i+2)*position[i]*scaleFactor[i]);
//         elevated[0] = elevated[1] + 2*position[0]*scaleFactor[0];
//
//         // Prepare to find the closest lattice points
//         double scale = 1.0f/(d+1);
//
//         // Greedily search for the closest remainder-zero lattice point
//         int sum = 0;
//         for (int i = 0; i <= d; i++) {
//             double v = elevated[i]*scale;
//             double up = ceilf(v)*(d+1);
//             double down = floorf(v)*(d+1);
//             if (up - elevated[i] < elevated[i] - down) {
//                 greedy[i] = (short)up;
//             } else {
//                 greedy[i] = (short)down;
//             }
//             sum += greedy[i];
//         }
//         sum /= d+1;
//
//         // Rank differential to find the permutation between this
//         // simplex and the canonical one.
//         for (int i = 0; i < d+1; i++) rank[i] = 0;
//         for (int i = 0; i < d; i++) {
//             for (int j = i+1; j <= d; j++) {
//                 if (elevated[i] - greedy[i] < elevated[j] - greedy[j]) {
//                     rank[i]++;
//                 } else {
//                     rank[j]++;
//                 }
//             }
//         }
//
//         if (sum > 0) {
//             // Sum too large - the point is off the hyperplane. We
//             // need to bring down the ones with the smallest
//             // differential
//             for (int i = 0; i <= d; i++) {
//                 if (rank[i] >= d + 1 - sum) {
//                     greedy[i] -= d+1;
//                     rank[i] += sum - (d+1);
//                 } else {
//                     rank[i] += sum;
//                 }
//             }
//         } else if (sum < 0) {
//             // Sum too small - the point is off the hyperplane. We
//             // need to bring up the ones with largest differential
//             for (int i = 0; i <= d; i++) {
//                 if (rank[i] < -sum) {
//                     greedy[i] += d+1;
//                     rank[i] += (d+1) + sum;
//                 } else {
//                     rank[i] += sum;
//                 }
//             }
//         }
//
//         // Compute barycentric coordinates
//         for (int i = 0; i < d+2; i++) { barycentric[i] = 0.0f; }
//         for (int i = 0; i <= d; i++) {
//             barycentric[d-rank[i]] += (elevated[i] - greedy[i]) * scale;
//             barycentric[d+1-rank[i]] -= (elevated[i] - greedy[i]) * scale;
//         }
//         barycentric[0] += 1.0f + barycentric[d+1];
//
//         // Splat the value into each vertex of the simplex, with
//         // barycentric weights
//         for (int remainder = 0; remainder <= d; remainder++) {
//             // Compute the location of the lattice point explicitly
//             // (all but the last coordinate - it’s redundant because
//             // they sum to zero)
//             for (int i = 0; i < d; i++) {
//                 key[i] = greedy[i] + canonical[remainder*(d+1) + rank[i]];
//             }
//
//             // Retrieve pointer to the value at this vertex
//             double *val = hashTable.lookup(key, true);
//
//             // Accumulate values with barycentric weight
//             for (int i = 0; i < vd; i++) {
//                 val[i] += barycentric[remainder]*value[i];
//             }
//
//             // Record this interaction to use later when slicing
//             replay[nReplay].offset = val - &hashTable.getValues()[0];
//             replay[nReplay].weight = barycentric[remainder];
//             nReplay++;
//         }
//     }
//
//     // Prepare for slicing
//     void beginSlice() {
//         nReplay = 0;
//     }
//
//     // Performs slicing out of position vectors. The barycentric
//     // weights and the simplex containing each position vector were
//     // calculated and stored in the splatting step.
//     void slice(double *col) {
//         const vector<double> &vals = hashTable.getValues();
//         for (int j = 0; j < vd; j++) { col[j] = 0; }
//         for (int i = 0; i <= d; i++) {
//             ReplayEntry r = replay[nReplay++];
//             for (int j = 0; j < vd; j++) {
//                 col[j] += r.weight*vals[r.offset + j];
//             }
//         }
//     }
//
//     // Performs a Gaussian blur along each projected axis in the hyperplane.
//     void blur() {
//
//         // Prepare temporary arrays
//         vector<short> neighbor1(d+1), neighbor2(d+1);
//         vector<double> zero(vd, 0.0f);
//         vector<double> newValue(vd*hashTable.size());
//         vector<double> &oldValue = hashTable.getValues();
//
//         // For each of d+1 axes,
//         for (int j = 0; j <= d; j++) {
//             // For each vertex in the lattice,
//             for (int i = 0; i < hashTable.size(); i++) {
//                 // Blur point i in dimension j
//                 short *key = &(hashTable.getKeys()[i*d]);
//                 for (int k = 0; k < d; k++) {
//                     neighbor1[k] = key[k] + 1;
//                     neighbor2[k] = key[k] - 1;
//                 }
//                 neighbor1[j] = key[j] - d;
//                 neighbor2[j] = key[j] + d;
//
//                 double *oldVal = &oldValue[i*vd];
//                 double *newVal = &newValue[i*vd];
//
//                 double *v1 = hashTable.lookup(neighbor1, false);
//                 double *v2 = hashTable.lookup(neighbor2, false);
//                 if (!v1) v1 = &zero[0];
//                 if (!v2) v2 = &zero[0];
//
//                 // Mix values of the three vertices
//                 for (int k = 0; k < vd; k++) {
//                     newVal[k] = (v1[k] + 2*oldVal[k] + v2[k]);
//                 }
//             }
//             newValue.swap(oldValue);
//         }
//     }
// private:
//     int d, vd, n;
//     vector<double> elevated, scaleFactor, barycentric;
//     vector<short> canonical, key, greedy;
//     vector<char> rank;
//
//     struct ReplayEntry {
//         int offset;
//         double weight;
//     };
//     vector<ReplayEntry> replay;
//     int nReplay;
//
//     HashTablePermutohedral hashTable;
//
// };
//
//
//
//
//
//
//
//
//
//
//
//
// // A bilateral filter of a color image with the given spatial standard
// // deviation and color-space standard deviation
// void bilateral(cv::Mat& im, double spatialSigma, double colorSigma) {
//
//     // Construct the five-dimensional position vectors and
//     // four-dimensional value vectors
//     vector<double> positions(im.cols*im.rows*5);
//     vector<double> values(im.cols*im.rows*4);
//     int idx = 0;
//
//     clock_t now;
// 	std::cout << "start filling positions and values" << std::endl;
//     now = clock();
//     printf( "now is %f seconds\n", (double)(now) / CLOCKS_PER_SEC);
//     for (int y = 0; y < im.cols; y++) {
//         for (int x = 0; x < im.rows; x++) {
//             positions[idx*5+0] = x/spatialSigma;
//             positions[idx*5+1] = y/spatialSigma;
//             positions[idx*5+2] = im.at<cv::Vec3b>(x,y)[0]/colorSigma;
//             positions[idx*5+3] = im.at<cv::Vec3b>(x,y)[1]/colorSigma;
//             positions[idx*5+4] = im.at<cv::Vec3b>(x,y)[2]/colorSigma;
//             values[idx*4+0] = im.at<cv::Vec3b>(x,y)[0];
//             values[idx*4+1] = im.at<cv::Vec3b>(x,y)[1];
//             values[idx*4+2] = im.at<cv::Vec3b>(x,y)[2];
//             values[idx*4+3] = 1.0f;
//             idx++;
//         }
//     }
//
// 	std::cout << "start PermutohedralLattice::filter" << std::endl;
//     now = clock();
//     printf( "now is %f seconds\n", (double)(now) / CLOCKS_PER_SEC);
//
//     // Perform the Gauss transform. For the five-dimensional case the
//     // Permutohedral Lattice is appropriate.
//     PermutohedralLattice::filter(&positions[0], 5,
//                                     &values[0], 4,
//                                     im.cols*im.rows,
//                                     &values[0]);
//
//     // Divide through by the homogeneous coordinate and store the
//     // result back to the image
//     idx = 0;
//     for (int y = 0; y < im.cols; y++) {
//         for (int x = 0; x < im.rows; x++) {
//             double w = values[idx*4+3];
//             im.at<cv::Vec3b>(x,y)[0] = values[idx*4+0]/w;
//             im.at<cv::Vec3b>(x,y)[1] = values[idx*4+1]/w;
//             im.at<cv::Vec3b>(x,y)[2] = values[idx*4+2]/w;
//             idx++;
//         }
//     }
// }
//
//
// // A bilateral filter of a color image with the given spatial standard
// // deviation and color-space standard deviation
// void bilateral(cv::Mat& im,cv::Mat& target, double spatialSigma, double colorSigma) {
//
//     if(im.cols != target.cols || im.rows != target.rows)
//     {
//         std::cout << "the shape of target is different from im " << std::endl;
//     }
//
//     // Construct the five-dimensional position vectors and
//     // four-dimensional value vectors
//     vector<double> positions(im.cols*im.rows*5);
//     vector<double> values(im.cols*im.rows*4);
//     int idx = 0;
//
//     clock_t now;
// 	std::cout << "start filling positions and values" << std::endl;
//     now = clock();
//     printf( "now is %f seconds\n", (double)(now) / CLOCKS_PER_SEC);
//     for (int y = 0; y < im.cols; y++) {
//         for (int x = 0; x < im.rows; x++) {
//             positions[idx*5+0] = x/spatialSigma;
//             positions[idx*5+1] = y/spatialSigma;
//             positions[idx*5+2] = im.at<cv::Vec3b>(x,y)[0]/colorSigma;
//             positions[idx*5+3] = im.at<cv::Vec3b>(x,y)[1]/colorSigma;
//             positions[idx*5+4] = im.at<cv::Vec3b>(x,y)[2]/colorSigma;
//             values[idx*4+0] = target.at<uchar>(x,y);
//             // values[idx*4+1] = target.at<uchar>(x,y);
//             // values[idx*4+2] = target.at<uchar>(x,y);
//             values[idx*4+1] = 1.0f;
//             values[idx*4+2] = 1.0f;
//             values[idx*4+3] = 1.0f;
//             idx++;
//         }
//     }
//
// 	std::cout << "start PermutohedralLattice::filter" << std::endl;
//     now = clock();
//     printf( "now is %f seconds\n", (double)(now) / CLOCKS_PER_SEC);
//
//     // Perform the Gauss transform. For the five-dimensional case the
//     // Permutohedral Lattice is appropriate.
//     PermutohedralLattice::filter(&positions[0], 5,
//                                     &values[0], 4,
//                                     im.cols*im.rows,
//                                     &values[0]);
//
//     // Divide through by the homogeneous coordinate and store the
//     // result back to the image
//     idx = 0;
//     for (int y = 0; y < im.cols; y++) {
//         for (int x = 0; x < im.rows; x++) {
//             double w = values[idx*4+3];
//             target.at<uchar>(x,y) = values[idx*4+0]/w;
//             // target.at<cv::uchar>(x,y) = values[idx*4+1]/w;
//             // target.at<cv::uchar>(x,y) = values[idx*4+2]/w;
//             // target.at<cv::Vec3b>(x,y)[0] = values[idx*4+0]/w;
//             // target.at<cv::Vec3b>(x,y)[1] = values[idx*4+0]/w;
//             // target.at<cv::Vec3b>(x,y)[2] = values[idx*4+0]/w;
//             idx++;
//         }
//     }
// }
//




std::vector<double> unique(const cv::Mat& input, bool sort = false)
{
    if (input.channels() > 1 || input.type() != CV_32F)
    {
        std::cerr << "unique !!! Only works with CV_32F 1-channel Mat" << std::endl;
        return std::vector<double>();
    }

    std::vector<double> out;
    for (int y = 0; y < input.rows; ++y)
    {
        const double* row_ptr = input.ptr<double>(y);
        for (int x = 0; x < input.cols; ++x)
        {
            double value = row_ptr[x];

            if ( std::find(out.begin(), out.end(), value) == out.end() )
                out.push_back(value);
        }
    }

    if (sort)
        std::sort(out.begin(), out.end());

    return out;
}
//
//
//
// // template<class ForwardIt, class T, class Compare=std::less<>>
// // ForwardIt binary_find(ForwardIt first, ForwardIt last, const T& value, Compare comp={})
// // {
// //     // Note: BOTH type T and the type after ForwardIt is dereferenced
// //     // must be implicitly convertible to BOTH Type1 and Type2, used in Compare.
// //     // This is stricter than lower_bound requirement (see above)
// //
// //     first = std::lower_bound(first, last, value, comp);
// //     return first != last && !comp(value, *first) ? first : last;
// // }

template<class ForwardIt, class T>
ForwardIt binary_find(ForwardIt first, ForwardIt last, const T& value)
{
    // Note: BOTH type T and the type after ForwardIt is dereferenced
    // must be implicitly convertible to BOTH Type1 and Type2, used in Compare.
    // This is stricter than lower_bound requirement (see above)

    first = std::lower_bound(first, last, value);
    return (first != last) && (value < *first) ? first : last;
}



class BilateralGrid
{
public:
    // Performs a Gauss transform
    // pos : position vectors
    // pd : position dimensions
    // val : value vectors
    // vd : value dimensions
    // n : number of items to filter
    // out : place to store the output
    static void solver(const double *pos, int pd,
                        const double *val, int vd,
                        int n, double *out) {


    }


    // Performs a Gauss transform
    // pos : position vectors
    // pd : position dimensions
    // val : value vectors
    // vd : value dimensions
    // n : number of items to filter
    // out : place to store the output
    static void filter(const double *pos, int pd,
                        const double *val, int vd,
                        int n, double *out) {

    }




    // BilateralGrid constructor
    // pd : dimensionality of position vectors
    // vd : dimensionality of value vectors
    // n : number of points in the input
    BilateralGrid(int pd, int vd, int n) :
        dim(pd), vd(vd), npixels(n) {

        // Allocate storage for various arrays
        // blurs.resize(dim);
        bs_param = bs_params();
        grid_param = grid_params();


    }


    void solve(std::vector<double>& x, int pd, std::vector<double>& w,int vd, int n,std::vector<double>& out) {

        compute_factorization(x);

        bistochastize(10);

        Eigen::SparseMatrix<double> bluredDn(nvertices,nvertices);
        Blur(Dn,bluredDn);
	    std::cout << "start Blur(Dn,bluredDn)" << std::endl;
        Eigen::SparseMatrix<double> A_smooth = Dm - Dn * bluredDn;
        // SparseMatrix<double> A_diag(nvertices);
        Eigen::SparseMatrix<double> M(nvertices,nvertices);
        Eigen::SparseMatrix<double> A_data(nvertices,nvertices);
        Eigen::SparseMatrix<double> A(nvertices,nvertices);
        Eigen::VectorXd b(nvertices);
        Eigen::VectorXd y(nvertices);
        std::vector<double> w_splat;
        std::vector<double> xw(x.size());
        std::vector<double> y0;
        std::vector<double> yhat;
        Splat(w,w_splat);
        diags(w_splat,A_data);
        A =  (A_smooth + A_data) * (bs_param.lam);
        for (int i = 0; i < x.size(); i++) {
            xw[i] = x[i] * w[i];
        }

        Splat(xw,b);

        for (int i = 0; i < nvertices; i++) {
            if(A.coeff(i,i) > bs_param.A_diag_min)
            {
                M.insert(i,i) = 1.0/A.coeff(i,i);
            }
            else
            {
                M.insert(i,i) = 1.0/bs_param.A_diag_min;
            }
        }

        for (int i = 0; i < b.size(); i++) {
            y0[i] = b(i) / w_splat[i];
        }
        yhat = y0;       // why shold empty_like(y0)

        // fill A and b
        Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower|Eigen::Upper> cg;
        cg.compute(A);
        for (size_t i = 0; i < bs_param.cg_maxiter; i++) {
            y = cg.solve(b);
            std::cout << "#iterations:     " << cg.iterations() << std::endl;
            std::cout << "estimated error: " << cg.error()      << std::endl;
        }

        Slice(y,out);




    }



    void Splat(std::vector<double>& x, std::vector<double>& result) {
        Eigen::SparseVector<double> v(x.size());
        for (int i = 0; i < x.size(); i++) {
            v.coeffRef(i) = x[i];
        }
        Eigen::SparseVector<double> vres(nvertices);
        vres = S*v;

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


    void bistochastize(int maxiter = 10) {
        std::vector<double> ones_npixels(npixels,1.0);
        std::vector<double> n(nvertices,1.0);
        std::vector<double> m;
        Splat(ones_npixels,m);

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


    //
    //
    // void unique(std::vector<double>& hashed_coords, std::vector<double>& unique_hashes,
    //             std::vector<int>& unique_idx,std::vector<int>& idx)
    // {
    //     std::cout << "for 1" << std::endl;
    //     for (int i = 0; i < hashed_coords.size(); i++) {
    //         if(std::find(unique_hashes.begin(), unique_hashes.end(), hashed_coords[i]) == unique_hashes.end())
    //             unique_hashes.push_back(hashed_coords[i]);
    //     }
    //     std::cout << "sort 1" << std::endl;
    //     std::sort(unique_hashes.begin(), unique_hashes.end());
    //
    //     std::cout << "for 2" << std::endl;
    //     for (int i = 0; i < hashed_coords.size(); i++) {
    //         std::vector<double>::iterator iter = std::find(unique_hashes.begin(), unique_hashes.end(), hashed_coords[i]);
    //         idx.push_back(std::distance(unique_hashes.begin(),iter));
    //     }
    //
    //     std::cout << "for 3" << std::endl;
    //     for (int i = 0; i < unique_hashes.size(); i++) {
    //         std::vector<double>::iterator iter = std::find(hashed_coords.begin(), hashed_coords.end(), unique_hashes[i]);
    //         unique_idx.push_back(std::distance(hashed_coords.begin(),iter));
    //     }
    //
    // }
    //



    void unique(std::vector<double>& hashed_coords, std::vector<double>& unique_hashes,
                std::vector<int>& unique_idx,std::vector<int>& idx)
    {
        std::set<double> input;
        std::cout << "for 1" << std::endl;
        for (int i = 0; i < hashed_coords.size(); i++) {
            input.insert(hashed_coords[i]);
        }
        unique_hashes.resize(input.size());
        unique_idx.resize(input.size(),-1);
        idx.resize(npixels);
        std::copy(input.begin(),input.end(),unique_hashes.begin());

        std::cout << "for 2" << std::endl;
        for (int i = 0; i < hashed_coords.size(); i++) {
            std::set<double>::iterator iter = input.find(hashed_coords[i]);
            idx.push_back(std::distance(input.begin(),iter));
            if(unique_idx[idx[idx.size()-1]] < 0) unique_idx[idx[idx.size()-1]] = i;
        }

        std::cout << "for 2 end" << std::endl;

    }


    void get_valid_idx(std::vector<double>& valid, std::vector<double>& candidates,
                        std::vector<int>& valid_idx, std::vector<int>& locs)
    {
        valid_idx.clear();
        locs.clear();
        for (int i = 0; i < candidates.size(); i++) {
            std::vector<double>::iterator iter = binary_find(valid.begin(), valid.end(), candidates[i]);
            if(iter != valid.end())
            {
                locs.push_back(std::distance(valid.begin(), iter));
                valid_idx.push_back(i);
            }
        }

    }

    void csr_matrix(Eigen::SparseMatrix<double>& spmat, std::vector<double>& values,
                    std::vector<int>& rows, std::vector<int>& cols)
    {
        for (int i = 0; i < values.size(); i++) {
            spmat.insert(rows[i],cols[i]) = values[i];
        }
    }

    void diags(std::vector<double>& v,Eigen::SparseMatrix<double>& m) {
        m = Eigen::SparseMatrix<double>(v.size(),v.size());
        for (int i = 0; i < v.size(); i++) {
            m.insert(i,i) = v[i];
        }
    }

    void hash_coords(std::vector<double>& coords_flat, std::vector<double>& hashed_coords)
    {
        double max_val = 255.0;
        hashed_coords.clear();
        for (int i = 0; i < npixels; i++) {
            double hash = 0;
            for (int j = 0; j < dim; j++) {
                hash += coords_flat[i*dim+j]*max_val;
            }
            hashed_coords.push_back(hash);
        }
    }

    void compute_factorization(std::vector<double>& coords_flat)
    {
        std::vector<double> hashed_coords;
        std::vector<double> unique_hashes;
        std::vector<double> unique_coords;
        std::vector<int> unique_idx;
        std::vector<int> idx;
        std::vector<double> ones_npixels(npixels,1.0);
        std::vector<int> arange_npixels(npixels);

        for (int i = 0; i < npixels; i++) {
            arange_npixels.push_back(i);
        }

        std::cout << "start hash_coords(coords_flat,hash_coords)" << std::endl;
        hash_coords(coords_flat,hashed_coords);

        unique(hashed_coords,unique_hashes,unique_idx,idx);
        std::cout << "finish unique()" << std::endl;

        nvertices = unique_idx.size();
        std::vector<double> ones_nvertices(nvertices,1.0);
        S = Eigen::SparseMatrix<double>(npixels,nvertices);
        for (int i = 0; i < nvertices; i++) {
            for (int j = 0; j < dim; j++) {
                unique_coords.push_back(coords_flat[unique_idx[i]*dim+j]);
            }
        }

        std::cout << "start Construct csr_matrix S" << std::endl;
        csr_matrix(S, ones_npixels, idx, arange_npixels);


        std::cout << "start Construct blurs" << std::endl;
        for (int i = 0; i < dim; i++) {
            Eigen::SparseMatrix<double> blur(nvertices,nvertices);
            for (int j = -1; j <= 1; j++) {
                Eigen::SparseMatrix<double> blur_temp(nvertices,nvertices);
                std::vector<double> neighbor_coords = unique_coords;
                std::vector<double> neighbor_hashes;
                std::vector<int> valid_coord;
                std::vector<int> neighbor_idx;
                std::vector<int> valid_idx;
                for (int k = 0; k < nvertices; k++) {
                    neighbor_coords[k*dim+i] += j;
                }
                hash_coords(neighbor_coords,neighbor_hashes);
                get_valid_idx(unique_hashes,neighbor_hashes,valid_coord,neighbor_idx);
                csr_matrix(blur_temp, ones_nvertices, valid_coord, neighbor_idx);
                blur = blur + blur_temp;
            }
            std::cout << "blur"<< i << std::endl;
            blurs.push_back(blur);
        }


    }






private:
    int npixels;
    int nvertices;
    int dim;
    int pd;
    int vd;
    // std::vector<double> hash_vec;
    std::vector<Eigen::SparseMatrix<double> > blurs;
    Eigen::SparseMatrix<double> S;
    Eigen::SparseMatrix<double> Dn;
    Eigen::SparseMatrix<double> Dm;

    struct grid_params
    {
        double spatialSigma;
        double lumaSigma;
        double chromaSigma;
        grid_params()
        {
            spatialSigma = 8.0;
            lumaSigma = 4.0;
            chromaSigma = 4.0;
        }
    };

    struct bs_params
    {
        double lam;
        double A_diag_min;
        double cg_tol;
        int cg_maxiter;
        bs_params()
        {
            lam = 128.0;
            A_diag_min = 1e-5;
            cg_tol = 1e-5;
            cg_maxiter = 25;
        }
    };

    grid_params grid_param;
    bs_params bs_param;


};



// A bilateral solver of a color image with the given spatial standard
// deviation and color-space standard deviation
void bilateral(cv::Mat& reference,cv::Mat& target, double spatialSigma, double lumaSigma, double chromaSigma)
{

    if(reference.cols != target.cols || reference.rows != target.rows)
    {
        std::cout << "the shape of target is different from reference " << std::endl;
    }

    // Construct the five-dimensional position vectors and
    // four-dimensional value vectors
    std::vector<double> positions(reference.cols*reference.rows*5);
    std::vector<double> values(reference.cols*reference.rows*4);
    int idx = 0;

    clock_t now;
	std::cout << "start filling positions and values" << std::endl;
    now = clock();
    printf( "now is %f seconds\n", (double)(now) / CLOCKS_PER_SEC);
    for (int y = 0; y < reference.cols; y++) {
        for (int x = 0; x < reference.rows; x++) {
            positions[idx*5+0] = x/spatialSigma;
            positions[idx*5+1] = y/spatialSigma;
            positions[idx*5+2] = reference.at<cv::Vec3b>(x,y)[0]/lumaSigma;
            positions[idx*5+3] = reference.at<cv::Vec3b>(x,y)[1]/chromaSigma;
            positions[idx*5+4] = reference.at<cv::Vec3b>(x,y)[2]/chromaSigma;
            values[idx*4+0] = target.at<uchar>(x,y);
            // values[idx*4+1] = target.at<uchar>(x,y);
            // values[idx*4+2] = target.at<uchar>(x,y);
            values[idx*4+1] = 1.0f;
            values[idx*4+2] = 1.0f;
            values[idx*4+3] = 1.0f;
            idx++;
        }
    }

	std::cout << "start BilateralGrid" << std::endl;
    now = clock();
    printf( "now is %f seconds\n", (double)(now) / CLOCKS_PER_SEC);

    BilateralGrid grid(5, 4, reference.cols*reference.rows);
	std::cout << "start BilateralGrid::solve" << std::endl;
    grid.solve(positions, 5, values, 4, reference.cols*reference.rows, values);

    // Divide through by the homogeneous coordinate and store the
    // result back to the image
    idx = 0;
    for (int y = 0; y < reference.cols; y++) {
        for (int x = 0; x < reference.rows; x++) {
            double w = values[idx*4+3];
            target.at<uchar>(x,y) = values[idx*4+0]/w;
            // target.at<cv::uchar>(x,y) = values[idx*4+1]/w;
            // target.at<cv::uchar>(x,y) = values[idx*4+2]/w;
            // target.at<cv::Vec3b>(x,y)[0] = values[idx*4+0]/w;
            // target.at<cv::Vec3b>(x,y)[1] = values[idx*4+0]/w;
            // target.at<cv::Vec3b>(x,y)[2] = values[idx*4+0]/w;
            idx++;
        }
    }
}








int main(int argc, char const *argv[]) {
    std::cout << "hello opencv" << '\n';
    cv::Mat im = cv::imread(argv[1]);
    cv::Mat im1 = cv::imread(argv[1]);
    cv::Mat target = cv::imread(argv[2]);
    // cv::Mat target = cv::imread(argv[2],0);

	cvtColor(im, im, cv::COLOR_BGR2YCrCb);
	cvtColor(target, target, cv::COLOR_BGR2YCrCb);

    // cv::Mat im1 = cv::imread("flower8.jpg");
    std::cout << "im:" << im.cols<<"x"<< im.rows<< std::endl;
    // cv::imshow("im",im);
    // cv::waitKey(1000);

    clock_t start, finish, now;
    double duration;

    // now = clock();
    // printf( "start BilateralGrid now is %f seconds\n", (double)(now) / CLOCKS_PER_SEC);
    // BilateralGrid gird(im);
        //创建窗口
    // cv::namedWindow("双边滤波【原图】");
    // cv::namedWindow("双边滤波【效果图】");

    //显示原图
    // cv::imshow("双边滤波【原图】", im);

    //进行双边滤波操作
    // cv::Mat out;
    // cv::bilateralFilter(im, out, 50, 50 * 2, 50 / 2);

    //显示效果图
    // cv::imshow("双边滤波【效果图】", out);

    // cv::waitKey(0);

    // now = clock();
    // printf( "end BilateralGrid now is %f seconds\n", (double)(now) / CLOCKS_PER_SEC);

    start = clock();
    now = clock();
    printf( "now is %f seconds\n", (double)(now) / CLOCKS_PER_SEC);
	// bilateral(im,8.0,4.0);
	bilateral(im, target, 16.0, 8.0, 8.0);
	// bilateral(im,64.0,32.0);
    finish = clock();
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    printf( "%f seconds\n", duration );
	// cv::Mat im2 = 2*(im1-im);
	// cv::imshow("output",im);

	cvtColor(im, im, cv::COLOR_YCrCb2BGR);
	cvtColor(target, target, cv::COLOR_YCrCb2BGR);


	cv::imshow("input",im1);
	cv::imshow("output",target);
	cv::waitKey(0);

    return 0;
}
