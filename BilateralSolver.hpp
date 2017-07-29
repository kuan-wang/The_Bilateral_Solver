

#ifndef _BILATERALSOLVER_HPP_
#define _BILATERALSOLVER_HPP_


#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>




#include<opencv2/core/core.hpp>
#include<opencv2/core/eigen.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/opencv.hpp>



#include <set>
#include <cmath>
#include <omp.h>
#include <math.h>
#include <time.h>
#include <vector>
#include <memory>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>





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
    BilateralGrid(std::vector<double>& reference, int pd, int vd, int n) :
        dim(pd), vd(vd), npixels(n) {

        // Allocate storage for various arrays
        // blurs.resize(dim);
        bs_param = bs_params();
        grid_param = grid_params();

        compute_factorization(reference);

        bistochastize(10);

    }

    BilateralGrid(Eigen::MatrixXd& reference, int pd, int vd, int n) :
        dim(pd), vd(vd), npixels(n) {

        // Allocate storage for various arrays
        // blurs.resize(dim);
        bs_param = bs_params();
        grid_param = grid_params();

        compute_factorization(reference);

        bistochastize(10);

    }

    // void filt(std::vector<double>& x, int pd, std::vector<double>& w,int vd, int n,std::vector<double>& out)
    // {
    //
    // }


    void filt(std::vector<double>& x, std::vector<double>& w,std::vector<double>& out)
    {

        for (int i = 0; i < npixels; i++)
        {
            x[i] = x[i] * w[i];
        }

        std::vector<double> spalt_result;
        std::vector<double> blur_result;
        std::vector<double> slice_result(npixels, -1);

        std::vector<double> spalt_wresult;
        std::vector<double> blur_wresult;
        std::vector<double> slice_wresult(npixels, -1);

        std::vector<double> onesx(npixels,1);
        std::vector<double> spalt_onesresult;
        std::vector<double> blur_onesresult;
        std::vector<double> slice_onesresult(npixels, -1);


        Splat(x, spalt_result);
        Splat(w, spalt_wresult);
        Splat(onesx, spalt_onesresult);
        // std::cout << "x:" << std::endl;
        // PrintVector(x);
        // std::cout << "spalt_result:" << std::endl;
        // PrintVector(spalt_result);

        Blur(spalt_result, blur_result);
        Blur(spalt_wresult, blur_wresult);
        Blur(spalt_onesresult, blur_onesresult);
        // std::cout << "blur_result:" << std::endl;
        // PrintVector(blur_result);

        Slice(blur_result, slice_result);
        Slice(blur_wresult, slice_wresult);
        Slice(blur_onesresult, slice_onesresult);
        // std::cout << "slice_result:" << std::endl;
        // PrintVector(slice_result);
        // std::cout << "slice_onesresult:" << std::endl;
        // PrintVector(slice_onesresult);

        out.resize(npixels);
        for (int i = 0; i < npixels; i++)
        {
            out[i] = slice_result[i]/slice_wresult[i];
            // out[i] = slice_result[i]/slice_onesresult[i];
            // out[i] = slice_wresult[i]/slice_onesresult[i];
        }


    }





    void solve(Eigen::MatrixXd& x,
               Eigen::MatrixXd& w,
               Eigen::MatrixXd& out)
    {

        clock_t now;

        Eigen::SparseMatrix<double> bluredDn(nvertices,nvertices);
        Blur(Dn,bluredDn);
	    // std::cout << "start Blur(Dn,bluredDn)" << std::endl;
        Eigen::SparseMatrix<double> A_smooth = Dm - Dn * bluredDn;
        now = clock();
        printf( "A_smooth : now is %f seconds\n\n", (double)(now) / CLOCKS_PER_SEC);


        // SparseMatrix<double> A_diag(nvertices);
        Eigen::SparseMatrix<double> M(nvertices,nvertices);
        Eigen::SparseMatrix<double> A_data(nvertices,nvertices);
        Eigen::SparseMatrix<double> A(nvertices,nvertices);
        Eigen::VectorXd b(nvertices);
        Eigen::VectorXd y(nvertices);
        Eigen::VectorXd w_splat(nvertices);
        Eigen::VectorXd xw(x.size());


	    // std::cout << "start Splat(w,w_splat)" << std::endl;
        Splat(w,w_splat);
        diags(w_splat,A_data);
        A = bs_param.lam * A_smooth + A_data ;
        now = clock();
        printf( "A : now is %f seconds\n\n", (double)(now) / CLOCKS_PER_SEC);

        xw = x.array() * w.array();

        Splat(xw,b);
        now = clock();
        printf( "b : now is %f seconds\n\n", (double)(now) / CLOCKS_PER_SEC);


        std::cout << "solve" << std::endl;
        // fill A and b
        Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower|Eigen::Upper> cg;
        cg.compute(A);
        for (size_t i = 0; i < bs_param.cg_maxiter; i++) {
            y = cg.solve(b);
            std::cout << "#iterations:     " << cg.iterations() << std::endl;
            std::cout << "estimated error: " << cg.error()      << std::endl;
            if(cg.error()  < bs_param.cg_tol) break;
        }
        now = clock();
        printf( "solved : now is %f seconds\n\n", (double)(now) / CLOCKS_PER_SEC);

        Slice(y,out);
        now = clock();
        printf( "Sliced : now is %f seconds\n\n", (double)(now) / CLOCKS_PER_SEC);



    }





//-------------------------------------------------------------------------------------
    void Splat(std::vector<double>& x, std::vector<double>& result) {
        Eigen::SparseVector<double> v(x.size());
        for (int i = 0; i < x.size(); i++) {
            v.coeffRef(i) = x[i];
        }
        // std::cout << "x.size:" << x.size() << std::endl;
        Eigen::SparseVector<double> vres(nvertices);
        result.resize(nvertices);
        // std::cout << "nvertices:" << nvertices << std::endl;
        vres = S * v;

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

    void Splat(Eigen::VectorXd& x, Eigen::VectorXd& result) {
        result = S*x;
    }

    void Splat(Eigen::MatrixXd& x, Eigen::VectorXd& result) {
        result = S*x;
    }

    void Splat(std::vector<double>& x, Eigen::SparseVector<double>& result) {
        Eigen::SparseVector<double> v(x.size());
        for (int i = 0; i < x.size(); i++) {
            v.coeffRef(i) = x[i];
        }
        result = S*v;
    }

    void Splat(Eigen::SparseVector<double>& x, Eigen::SparseVector<double>& result) {
        result = S*x;
    }

//-------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------

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

    void Slice(Eigen::SparseVector<double>& y, std::vector<double>& result) {
        Eigen::SparseVector<double> vres(nvertices);
        vres = S.transpose()*y;

        for(int i = 0; i < vres.size();i++)
        {
            result[i] = vres.coeff(i);
        }
    }

    void Slice(Eigen::VectorXd& y,
               Eigen::MatrixXd& result)
    {
        result = S.transpose()*y;
    }
//-------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------


    void Blur(Eigen::VectorXd& x, Eigen::VectorXd& result)
    {
        if(x.size() != nvertices)
        {
            std::cout << "x.size() != nvertices" << std::endl;
            exit(-1);
        }

        result = (2*dim)*x;
        for (int i = 0; i < dim; i++)
        {
            result = result + blurs[i] * x;
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




//-------------------------------------------------------------------------------------



    void bistochastize(int maxiter = 2)
    {
        Eigen::VectorXd ones_npixels = Eigen::VectorXd::Ones(npixels);
        Eigen::VectorXd n = Eigen::VectorXd::Ones(nvertices);
        Eigen::VectorXd m(nvertices);
        Eigen::VectorXd bluredn(nvertices);
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



    int binarySearchRecursive(double a[],int low,int high,double key){
        if(low>high)
            return -(low+1);

        int mid=low+(high-low)/2;
        if(key<a[mid])
            return binarySearchRecursive(a,low,mid-1,key);
        else if(key > a[mid])
            return binarySearchRecursive(a,mid+1,high,key);
        else
            return mid;

    }


    void unique(Eigen::MatrixXd& coords_flat, Eigen::MatrixXd& unique_coords,
                Eigen::VectorXd& hashed_coords, std::vector<double>& unique_hashes)
    {
        unique_hashes.clear();
        std::set<double> input;
        clock_t now;
        now = clock();
        printf( "start input set : now is %f seconds\n\n", (double)(now) / CLOCKS_PER_SEC);
        for (int i = 0; i < hashed_coords.size(); i++) {
            input.insert(hashed_coords(i));
        }

        now = clock();
        printf( "start resize unique_coords : now is %f seconds\n\n", (double)(now) / CLOCKS_PER_SEC);
        unique_coords.resize(input.size(),coords_flat.cols());
        nvertices = input.size();
        unique_hashes.resize(input.size());
        S = Eigen::SparseMatrix<double>(nvertices,npixels);

        now = clock();
        printf( "start std::copy : now is %f seconds\n\n", (double)(now) / CLOCKS_PER_SEC);
        std::copy(input.begin(),input.end(),unique_hashes.begin());

        now = clock();
        printf( "start construct S : now is %f seconds\n\n", (double)(now) / CLOCKS_PER_SEC);
        for (int i = 0; i < hashed_coords.size(); i++) {
            int id = binarySearchRecursive(&unique_hashes[0],0,unique_hashes.size()-1,hashed_coords(i));  //size()-1?
            // std::set<double>::iterator got = unique_hashes.find (hashed_coords(i));
            // if(got != unique_hashes.end())
            if(id >= 0)
            {
                S.insert(id, i) = 1.0;
                unique_coords.row(id) = coords_flat.row(i);
                // std::cout << "(id,i) : (" << id <<","<< i <<")"<< std::endl;
            }
        }
        now = clock();
        printf( "end construct S : now is %f seconds\n\n", (double)(now) / CLOCKS_PER_SEC);
    }


    void unique(std::vector<double>& hashed_coords, std::unordered_set<double>& unique_hashes,
                std::vector<int>& unique_idx, std::vector<int>& idx)
    {
        unique_idx.clear();
        idx.clear();
        std::cout << "for 1" << std::endl;
        std::cout << "hashed_coords size" <<hashed_coords.size()<< std::endl;
        for (int i = 0; i < hashed_coords.size(); i++) {
            unique_hashes.insert(hashed_coords[i]);
        }
        unique_idx.resize(unique_hashes.size(),-1);
        // idx.resize(npixels);
        std::cout << "unique_hashes size" <<unique_hashes.size()<< std::endl;

        std::cout << "for 2" << std::endl;
        int count = 0;
        for (int i = 0; i < hashed_coords.size(); i++) {
            // int id = binarySearchRecursive(&unique_hashes[0],0,input.size(),hashed_coords[i]);
            std::unordered_set<double>::iterator got = unique_hashes.find (hashed_coords[i]);
            if(got != unique_hashes.end())
            {
                int id = std::distance(unique_hashes.begin(), got);
                idx[count++] = id;
                if(unique_idx[id] < 0) unique_idx[id] = i;
            }
        }

        std::cout << "for 2 end" << std::endl;

    }

    void unique(std::vector<double>& hashed_coords, std::vector<double>& unique_hashes,
                std::vector<int>& unique_idx,std::vector<int>& idx)
    {

        unique_idx.clear();
        idx.clear();
        unique_hashes.clear();


        std::set<double> input;
        std::cout << "for 1" << std::endl;
        std::cout << "hashed_coords size" <<hashed_coords.size()<< std::endl;
        for (int i = 0; i < hashed_coords.size(); i++) {
            // std::cout << "hashed_coords:"<<hashed_coords[i] << std::endl;
            input.insert(hashed_coords[i]);
        }
        unique_hashes.resize(input.size());
        unique_idx.resize(input.size(),-1);
        idx.resize(hashed_coords.size());
        std::copy(input.begin(),input.end(),unique_hashes.begin());
        // std::cout << "input :" <<unique_hashes<< std::endl;
        std::cout << "input size" <<input.size()<< std::endl;

        std::cout << "for 2" << std::endl;
        clock_t now;
        now = clock();
        printf( "start test for 2 : now is %f seconds\n\n", (double)(now) / CLOCKS_PER_SEC);
        int count = 0;
        for (int i = 0; i < hashed_coords.size(); i++) {
            int id = binarySearchRecursive(&unique_hashes[0],0,input.size(),hashed_coords[i]);
            if(id >= 0)
            {
                // idx.push_back(id);
                idx[count++] = id;
                if(unique_idx[id] < 0) unique_idx[id] = i;
            }
        }
        now = clock();
        printf( "end test for 2 : now is %f seconds\n\n", (double)(now) / CLOCKS_PER_SEC);
        std::cout << "for 2 end" << std::endl;

    }


    // void get_valid_idx(std::vector<double>& valid, std::vector<double>& candidates,
    //                     std::vector<int>& valid_idx, std::vector<int>& locs)
    // {
    //     valid_idx.clear();
    //     locs.clear();
    //     for (int i = 0; i < candidates.size(); i++) {
    //         int id = binarySearchRecursive(&valid[0],0,candidates.size(),candidates[i]);
    //         if(id >= 0)
    //         {
    //             locs.push_back(id);
    //             valid_idx.push_back(i);
    //         }
    //     }
    //     // std::cout << "candidates.size()" << candidates.size() << std::endl;
    //     // std::cout << "valid_idx.size():"<< valid_idx.size() << std::endl;
    //
    // }


    void get_valid_idx(std::vector<double>& valid, Eigen::VectorXd& candidates,
                        Eigen::SparseMatrix<double>& blur_temp)
    {

        for (int i = 0; i < candidates.size(); i++)
        {

            int id = binarySearchRecursive(&valid[0],0,valid.size()-1,candidates(i));   //size()-1?
            if(id >= 0)
            {
                // std::cout << "(i,id) : (" << i <<","<< id <<")"<< std::endl;
                blur_temp.insert(id, i) = 1;
                // blur_temp.insert(i, id) = 1;
                // std::cout << "(i,id) : (" << i <<","<< id <<")"<< std::endl;
            }
        }
    }


    void get_valid_idx(std::vector<double>& valid, std::vector<double>& candidates,
                        std::vector<int>& valid_idx, std::vector<int>& locs)
    {
        valid_idx.clear();
        locs.clear();
        for (int i = 0; i < candidates.size(); i++)
        {
            int id = binarySearchRecursive(&valid[0],0,valid.size(),candidates[i]);
            if(id >= 0)
            {
                locs.push_back(id);
                valid_idx.push_back(i);
            }
        }
        // std::cout << "candidates.size()" << candidates.size() << std::endl;
        // std::cout << "valid_idx.size():"<< valid_idx.size() << std::endl;

    }


    void csr_matrix(Eigen::SparseMatrix<double>& spmat, std::vector<double>& values,
                    std::vector<int>& rows, std::vector<int>& cols)
    {
        for (int i = 0; i < values.size(); i++) {
            spmat.insert(rows[i],cols[i]) = values[i];
        }
    }

    void csr_matrix(Eigen::SparseMatrix<double>& spmat, std::vector<int>& rows, std::vector<int>& cols)
    {
        for (int i = 0; i < rows.size(); i++) {
            spmat.insert(rows[i],cols[i]) = 1;
        }
    }


    void diags(Eigen::VectorXd& v,Eigen::SparseMatrix<double>& m)
    {
        m = Eigen::SparseMatrix<double>(v.size(),v.size());
        for (int i = 0; i < v.size(); i++) {
            m.insert(i,i) = v(i);
        }
    }

    void diags(std::vector<double>& v,Eigen::SparseMatrix<double>& m)
    {
        m = Eigen::SparseMatrix<double>(v.size(),v.size());
        for (int i = 0; i < v.size(); i++) {
            m.insert(i,i) = v[i];
        }
    }


    void hash_coords(Eigen::MatrixXd& coords_flat, Eigen::VectorXd& hashed_coords)
    {
        double max_val = 255.0;
        Eigen::VectorXd hash_vec(5);
        hash_vec[0] = 1;
        for (int i = 1; i < 5; i++) {
            hash_vec[i] = hash_vec[i-1]*max_val;
        }
        hashed_coords = coords_flat * hash_vec;
    }


    void hash_coords(std::vector<double>& coords_flat, std::vector<double>& hashed_coords)
    {
        double max_val = 255.0;
        hashed_coords.clear();
        // hashed_coords.resize(coords_flat.size()/dim);
        for (int i = 0; i < coords_flat.size()/dim; i++) {
            double hash = 0;
            for (int j = 0; j < dim; j++) {
                hash = coords_flat[i*dim+j] + hash*max_val;
                // std::cout << "hash:"<< hash << std::endl;
            }
            hashed_coords.push_back(hash);
            // hashed_coords[i] = hash;
        }
        std::cout << "hashed_coords == nvertices" <<hashed_coords.size()<<nvertices << std::endl;
        std::cout << "coords_flat.size()/dim"<<coords_flat.size()/dim << std::endl;
    }



    void compute_factorization(Eigen::MatrixXd& coords_flat)
    {
        Eigen::VectorXd hashed_coords;
        Eigen::MatrixXd unique_coords;
        std::vector<double> unique_hashes;

        clock_t now;
        now = clock();
        printf( "start hashcoords : now is %f seconds\n\n", (double)(now) / CLOCKS_PER_SEC);
        hash_coords(coords_flat,hashed_coords);
        // std::cout << "coords_flat:" << std::endl;
        // std::cout << coords_flat << std::endl;
        // std::cout << "hashed_coords:" << std::endl;
        // std::cout << hashed_coords << std::endl;

        now = clock();
        printf( "start unique : now is %f seconds\n\n", (double)(now) / CLOCKS_PER_SEC);
        unique(coords_flat, unique_coords, hashed_coords, unique_hashes);
        std::cout << "finish unique()" << std::endl;

        // std::cout << "unique_coords:" << std::endl;
        // std::cout << unique_coords << std::endl;
        // std::cout << "unique_hashes:" << std::endl;
        // std::set<double>::iterator iter=unique_hashes.begin();
        // while(iter!=unique_hashes.end())
        // {
        //     std::cout<<*iter<<std::endl;
        //     ++iter;
        // }


        now = clock();
        printf( "start construct blur : now is %f seconds\n\n", (double)(now) / CLOCKS_PER_SEC);
        for (int i = 0; i < dim; i++) {
            Eigen::SparseMatrix<double> blur(nvertices,nvertices);
            for (int j = -1; j <= 1; j++) {
                if(j == 0) continue;
                Eigen::SparseMatrix<double> blur_temp(nvertices,nvertices);
                Eigen::MatrixXd neighbor_coords = unique_coords;
                Eigen::VectorXd neighbor_hashes;

                // std::vector<int> valid_coord;
                // std::vector<int> neighbor_idx;
                // std::vector<int> valid_idx;
                for (int k = 0; k < nvertices; k++) {
                    neighbor_coords(k,i) += j;
                }
                hash_coords(neighbor_coords,neighbor_hashes);
            // std::cout << "neighbor_coords:" << std::endl;
            // std::cout << neighbor_coords << std::endl;
                get_valid_idx(unique_hashes, neighbor_hashes, blur_temp);
                // get_valid_idx(unique_hashes,neighbor_hashes,valid_coord,neighbor_idx);
                // std::cout <<i<<j<< "nvertices,valid_coord.size,neighbor_idx.size:"<< nvertices<<valid_coord.size()<<neighbor_idx.size() << std::endl;
            // std::cout << "ones_valid_coord:" << std::endl;
            // PrintVector(ones_valid_coord);
            // std::cout << "valid_coord:" << std::endl;
            // PrintVector(valid_coord);
            // std::cout << "neighbor_idx:" << std::endl;
            // PrintVector(neighbor_idx);
            //     std::cout << "blur_temp:"<< blur_temp << std::endl;
                blur = blur + blur_temp;
                // std::cout << blur << std::endl;
            }
            // std::cout << "blur:"<< blur << std::endl;
            std::cout << "blur"<< i << std::endl;
            blurs.push_back(blur);
        }


    }


    void compute_factorization(std::vector<double>& coords_flat)
    {
        std::vector<double> hashed_coords;
        std::vector<double> unique_hashes;
        std::vector<double> unique_coords;
        std::vector<int> unique_idx;
        std::vector<int> idx;
        // std::vector<double> ones_npixels(npixels,1.0);
        std::vector<int> arange_npixels(npixels);
        // std::vector<int> test_npixels(npixels*30);

        clock_t now;
        now = clock();
        printf( "start test for : now is %f seconds\n\n", (double)(now) / CLOCKS_PER_SEC);
    // #pragma omp parallel for num_threads(4)
    // #pragma omp parallel for
        for (int i = 0; i < npixels; i++) {
            arange_npixels[i] = i;
        }
    // // #pragma omp parallel for num_threads(4)
    // // #pragma omp parallel for
    //     for (int i = 0; i < npixels*10; i++) {
    //         test_npixels[i] = i;
    //     }
    // // #pragma omp parallel for num_threads(4)
    // // #pragma omp parallel for
    //     for (int i = 0; i < npixels*10; i++) {
    //         test_npixels[i] = i;
    //     }
    // // #pragma omp parallel for num_threads(4)
    // // #pragma omp parallel for
    //     for (int i = 0; i < npixels*10; i++) {
    //         test_npixels[i] = i;
    //     }

        now = clock();
        printf( "end for : now is %f seconds\n\n", (double)(now) / CLOCKS_PER_SEC);

        std::cout << "start hash_coords(coords_flat,hash_coords)" << std::endl;
        hash_coords(coords_flat,hashed_coords);

        unique(hashed_coords,unique_hashes,unique_idx,idx);
        std::cout << "finish unique()" << std::endl;

        // std::cout << "hashed_coords:" << std::endl;
        // PrintVector(hashed_coords);
        // std::cout << "unique_hashes:" << std::endl;
        // PrintVector(unique_hashes);
        // std::cout << "unique_idx:" << std::endl;
        // PrintVector(unique_idx);
        // std::cout << "idx:" << std::endl;
        // PrintVector(idx);


        nvertices = unique_idx.size();
        S = Eigen::SparseMatrix<double>(nvertices,npixels);

        std::cout << "start Construct csr_matrix S" << std::endl;
        // csr_matrix(S, ones_npixels, idx, arange_npixels);
        csr_matrix(S, idx, arange_npixels);


        unique_coords.resize(nvertices*dim);
        for (int i = 0; i < nvertices; i++) {
            for (int j = 0; j < dim; j++) {
                unique_coords[i*dim+j] = (coords_flat[unique_idx[i]*dim+j]);
            }
        }

        std::cout << "start Construct blurs" << std::endl;
        for (int i = 0; i < dim; i++) {
            Eigen::SparseMatrix<double> blur(nvertices,nvertices);
            for (int j = -1; j <= 1; j++) {
                if(j == 0) continue;
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
            // std::cout << "neighbor_coords:" << std::endl;
            // PrintVector(neighbor_coords);
            // std::cout << "neighbor_hashes:" << std::endl;
            // PrintVector(neighbor_hashes);
                get_valid_idx(unique_hashes,neighbor_hashes,valid_coord,neighbor_idx);
                // std::vector<double> ones_valid_coord(valid_coord.size(),1.0);
                std::cout <<i<<j<< "nvertices,valid_coord.size,neighbor_idx.size:"<< nvertices<<" "<<valid_coord.size()<<" "<<neighbor_idx.size() << std::endl;
                csr_matrix(blur_temp, valid_coord, neighbor_idx);
            // std::cout << "ones_valid_coord:" << std::endl;
            // PrintVector(ones_valid_coord);
            // std::cout << "valid_coord:" << std::endl;
            // PrintVector(valid_coord);
            // std::cout << "neighbor_idx:" << std::endl;
            // PrintVector(neighbor_idx);
                // std::cout << "blur_temp:"<< blur_temp << std::endl;
                blur = blur + blur_temp;
            }
            // std::cout << "blur:"<< blur << std::endl;
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
void bilateral(cv::Mat& reference,cv::Mat& target, cv::Mat& confidence, double spatialSigma = 8.0, double lumaSigma = 4.0, double chromaSigma = 4.0)
{

    if(reference.cols != target.cols || reference.rows != target.rows)
    {
        std::cout << "the shape of target is different from reference " << std::endl;
    }

	cvtColor(reference, reference, cv::COLOR_BGR2YCrCb);
	cvtColor(target, target, cv::COLOR_BGR2YCrCb);
	cvtColor(confidence, confidence, cv::COLOR_BGR2YCrCb);



        cv::Mat r(reference.cols*reference.rows, 5, CV_64F);
        cv::Mat t(reference.cols*reference.rows, 1, CV_64F);
        cv::Mat c(reference.cols*reference.rows, 1, CV_64F);
        // std::vector<double> re(reference.cols*reference.rows*5);
        // std::vector<double> ta(reference.cols*reference.rows);
        // std::vector<double> co(reference.cols*reference.rows);
        int idx = 0;
    	std::cout << "start filling positions and values" << std::endl;
        clock_t now;
        now = clock();
        printf( "fill positions and values : now is %f seconds\n", (double)(now) / CLOCKS_PER_SEC);
        for (int y = 0; y < reference.cols; y++) {
            for (int x = 0; x < reference.rows; x++) {
                double *datar = r.ptr<double>(idx);
                datar[0] = ceilf(x/spatialSigma);
                datar[1] = ceilf(y/spatialSigma);
                datar[2] = ceilf(reference.at<cv::Vec3b>(x,y)[0]/lumaSigma);
                datar[3] = ceilf(reference.at<cv::Vec3b>(x,y)[1]/chromaSigma);
                datar[4] = ceilf(reference.at<cv::Vec3b>(x,y)[2]/chromaSigma);
                // datar[3] = 1.0;
                // datar[4] = 1.0;
                idx++;
            }
        }
        idx = 0;
        for (int y = 0; y < reference.cols; y++) {
            for (int x = 0; x < reference.rows; x++) {
                double *datac = c.ptr<double>(idx);
                // datac[0] = 1.0;
                datac[0] = confidence.at<cv::Vec3b>(x,y)[0];
                idx++;
            }
        }
        idx = 0;
        for (int y = 0; y < reference.cols; y++) {
            for (int x = 0; x < reference.rows; x++) {
                double *datat = t.ptr<double>(idx);
                datat[0] = target.at<cv::Vec3b>(x,y)[0];
                idx++;
            }
        }

        std::cout << "cv2eigen" << std::endl;
        Eigen::MatrixXd ref;
        Eigen::MatrixXd tar;
        Eigen::MatrixXd con;

        cv::cv2eigen(r,ref);
        cv::cv2eigen(t,tar);
        cv::cv2eigen(c,con);
        std::cout << "finished cv2eigen" << std::endl;






	std::cout << "start BilateralGrid" << std::endl;
    now = clock();
    printf( "now is %f seconds\n", (double)(now) / CLOCKS_PER_SEC);

    BilateralGrid grid(ref,5, 1, reference.cols*reference.rows);
	std::cout << "start BilateralGrid::solve" << std::endl;
    grid.solve(tar, con, tar);

    // Divide through by the homogeneous coordinate and store the
    // result back to the image
    idx = 0;
    for (int y = 0; y < reference.cols; y++) {
        for (int x = 0; x < reference.rows; x++) {
            // double w = values[idx*4+3];
            target.at<cv::Vec3b>(x,y)[0] = tar(idx);
            // target.at<cv::uchar>(x,y) = values[idx*4+1]/w;
            // target.at<cv::uchar>(x,y) = values[idx*4+2]/w;
            // target.at<cv::Vec3b>(x,y)[0] = values[idx*4+0]/w;
            // target.at<cv::Vec3b>(x,y)[1] = values[idx*4+0]/w;
            // target.at<cv::Vec3b>(x,y)[2] = values[idx*4+0]/w;
            idx++;
        }
    }


	// cvtColor(reference, reference, cv::COLOR_YCrCb2BGR);
	cvtColor(target, target, cv::COLOR_YCrCb2BGR);


}





void bilateral(cv::Mat& reference,cv::Mat& target, double spatialSigma = 8.0, double lumaSigma = 4.0)
{




    	cvtColor(reference, reference, cv::COLOR_BGR2YCrCb);
    	cvtColor(target, target, cv::COLOR_BGR2YCrCb);
    	// cvtColor(confidence, confidence, cv::COLOR_BGR2YCrCb

        std::cout << "reference:" << reference.cols<<"x"<< reference.rows<< std::endl;


        int npixels = reference.cols*reference.rows;

        cv::Mat r(npixels, 5, CV_64F);
        cv::Mat tu(npixels, 1, CV_64F);
        cv::Mat tv(npixels, 1, CV_64F);
        cv::Mat cu(npixels, 1, CV_64F);
        cv::Mat cv(npixels, 1, CV_64F);
        // std::vector<double> re(reference.cols*reference.rows*5);
        // std::vector<double> ta(reference.cols*reference.rows);
        // std::vector<double> co(reference.cols*reference.rows);
        int idx = 0;
    	std::cout << "start filling positions and values" << std::endl;
        clock_t now;
        now = clock();
        printf( "fill positions and values : now is %f seconds\n", (double)(now) / CLOCKS_PER_SEC);
        for (int y = 0; y < reference.cols; y++) {
            for (int x = 0; x < reference.rows; x++) {
                double *datar = r.ptr<double>(idx);
                datar[0] = ceilf(x/spatialSigma);
                datar[1] = ceilf(y/spatialSigma);
                datar[2] = ceilf(reference.at<cv::Vec3b>(x,y)[0]/lumaSigma);
                // datar[3] = ceilf(reference.at<cv::Vec3b>(x,y)[1]/chromaSigma);
                // datar[4] = ceilf(reference.at<cv::Vec3b>(x,y)[2]/chromaSigma);
                datar[3] = 1.0;
                datar[4] = 1.0;
                idx++;
            }
        }
        idx = 0;
        for (int y = 0; y < reference.cols; y++) {
            for (int x = 0; x < reference.rows; x++) {
                double *datac = cu.ptr<double>(idx);
                if(target.at<cv::Vec3b>(x,y)[1] == 128) datac[0] = 0;
                else datac[0] = 255;
                idx++;
            }
        }
        idx = 0;
        for (int y = 0; y < reference.cols; y++) {
            for (int x = 0; x < reference.rows; x++) {
                double *datac = cv.ptr<double>(idx);
                if(target.at<cv::Vec3b>(x,y)[2] == 128) datac[0] = 0;
                else datac[0] = 255;
                idx++;
            }
        }
        idx = 0;
        for (int y = 0; y < reference.cols; y++) {
            for (int x = 0; x < reference.rows; x++) {
                double *datat = tu.ptr<double>(idx);
                datat[0] = target.at<cv::Vec3b>(x,y)[1];
                idx++;
            }
        }
        idx = 0;
        for (int y = 0; y < reference.cols; y++) {
            for (int x = 0; x < reference.rows; x++) {
                double *datat = tv.ptr<double>(idx);
                datat[0] = target.at<cv::Vec3b>(x,y)[2];
                idx++;
            }
        }

        std::cout << "cv2eigen" << std::endl;
        Eigen::MatrixXd ref;
        Eigen::MatrixXd taru;
        Eigen::MatrixXd tarv;
        Eigen::MatrixXd conu;
        Eigen::MatrixXd conv;

        cv::cv2eigen(r,ref);
        cv::cv2eigen(tu,taru);
        cv::cv2eigen(tv,tarv);
        cv::cv2eigen(cu,conu);
        cv::cv2eigen(cv,conv);
        std::cout << "finished cv2eigen" << std::endl;


    	std::cout << "start BilateralGrid" << std::endl;
        now = clock();
        printf( "now is %f seconds\n", (double)(now) / CLOCKS_PER_SEC);
        BilateralGrid grid(ref,5, 1, reference.cols*reference.rows);

        now = clock();
        printf( "solve :now is %f seconds\n\n", (double)(now) / CLOCKS_PER_SEC);
        grid.solve(taru,conu,taru);
        grid.solve(tarv,conv,tarv);
        now = clock();
        printf( "solved :now is %f seconds\n\n", (double)(now) / CLOCKS_PER_SEC);

        // Divide through by the homogeneous coordinate and store the
        // result back to the image
        idx = 0;
        for (int y = 0; y < reference.cols; y++) {
            for (int x = 0; x < reference.rows; x++) {
                // double w = values[idx*4+3];
                target.at<cv::Vec3b>(x,y)[1] = taru(idx);
                target.at<cv::Vec3b>(x,y)[2] = tarv(idx);
                // target.at<cv::uchar>(x,y) = values[idx*4+1]/w;
                // target.at<cv::uchar>(x,y) = values[idx*4+2]/w;
                // target.at<cv::Vec3b>(x,y)[0] = values[idx*4+0]/w;
                // target.at<cv::Vec3b>(x,y)[1] = values[idx*4+0]/w;
                // target.at<cv::Vec3b>(x,y)[2] = values[idx*4+0]/w;
                idx++;
            }
        }

    	cvtColor(target, target, cv::COLOR_YCrCb2BGR);


}















#endif //_BILATERALSOLVER_HPP_
