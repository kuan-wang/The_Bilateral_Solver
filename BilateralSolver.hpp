

#ifndef _FastBilateralSolverFilterImpl_HPP_
#define _FastBilateralSolverFilterImpl_HPP_



#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>




#include<opencv2/core/core.hpp>
#include<opencv2/core/eigen.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>



#include <cmath>
#include <chrono>
#include <vector>
#include <memory>
#include <stdlib.h>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <unordered_map>

namespace cv
{
namespace ximgproc
{

class CV_EXPORTS_W FastBilateralSolverFilter : public Algorithm
{
public:
    CV_WRAP virtual void filter(InputArray src, InputArray confidence, OutputArray dst) = 0;
};

// CV_EXPORTS_W Ptr<FastBilateralSolverFilter> createFastBilateralSolverFilter(InputArray guide, double sigma_spatial = 8.0f, double sigma_luma = 8.0f, double sigma_chroma = 8.0f);

// CV_EXPORTS_W void fastBilateralSolverFilter(InputArray guide, InputArray src, InputArray confidence, OutputArray dst, double sigma_spatial, double sigma_luma, double sigma_chroma);


CV_EXPORTS_W
Ptr<FastBilateralSolverFilter> createFastBilateralSolverFilter(InputArray guide, double sigma_spatial = 8.0f, double sigma_luma = 8.0f, double sigma_chroma = 8.0f)
{
    return Ptr<FastBilateralSolverFilter>(FastBilateralSolverFilterImpl::create(guide, sigma_spatial, sigma_luma, sigma_chroma);
}

CV_EXPORTS_W
void fastBilateralSolverFilter(InputArray guide, InputArray src, InputArray confidence, OutputArray dst, double sigma_spatial, double sigma_luma, double sigma_chroma)
{
    Ptr<FastBilateralSolverFilter> fbs = createFastBilateralSolverFilter(guide, sigma_spatial, sigma_luma, sigma_chroma);
    fbs->filter(src, confidence, dst);
}



    class FastBilateralSolverFilterImpl : public FastBilateralSolverFilter
    {
    public:

        static Ptr<FastBilateralSolverFilterImpl> create(InputArray guide, double sigma_spatial, double sigma_luma, double sigma_chroma)
        {
            FastBilateralSolverFilterImpl *fbs = new FastBilateralSolverFilterImpl();
            fbs->init(guide,sigma_spatial,sigma_luma,sigma_chroma);
            return Ptr<FastBilateralSolverFilterImpl>(fbs);
        }

        // FastBilateralSolverFilterImpl(){}

        static void filter(InputArray& reference, InputArray& target, InputArray& confidence, OutputArray& output,
                              float sigma_spatial = 8, float sigma_luma = 4, float sigma_chroma = 4)
        {
            CV_Assert( reference.type() == CV_8UC3 && target.type() == CV_8UC1 && confidence.type() == CV_8UC1 \
                       && reference.size() == target.size() && reference.size() == confidence.size() );

            output.create(target.size(), target.type());
            Mat ref = reference.getMat();
            Mat tar = target.getMat();
            Mat con = confidence.getMat();
            Mat out = output.getMat();

            FastBilateralSolverFilterImpl bs;
            bs.init(ref, sigma_spatial, sigma_luma, sigma_chroma);
            // filter(target,confidence,output);
            bs.solve(tar,con,out);
        }

        void filt(cv::Mat& target, cv::Mat& confidence, cv::Mat& output);
        void solve(cv::Mat& target, cv::Mat& confidence, cv::Mat& output);
        void init(cv::Mat& reference_bgr, double sigma_spatial, double sigma_luma, double sigma_chroma);

        void Splat(Eigen::VectorXf& input, Eigen::VectorXf& output);
        void Blur(Eigen::VectorXf& input, Eigen::VectorXf& output);
        void Slice(Eigen::VectorXf& input, Eigen::VectorXf& output);

    private:

        int npixels;
        int nvertices;
        int dim;
        int pd;
        int vd;
        std::vector<Eigen::SparseMatrix<float, Eigen::ColMajor> > blurs;
        std::vector<int> splat_idx;
        std::vector<std::pair<int, int>> blur_idx;
        Eigen::VectorXf m;
        Eigen::VectorXf n;
        Eigen::SparseMatrix<float, Eigen::ColMajor> blurs_test;
        Eigen::SparseMatrix<float, Eigen::ColMajor> S;
        Eigen::SparseMatrix<float, Eigen::ColMajor> Dn;
        Eigen::SparseMatrix<float, Eigen::ColMajor> Dm;

          struct grid_params
          {
              float spatialSigma;
              float lumaSigma;
              float chromaSigma;
              grid_params()
              {
                  spatialSigma = 8.0;
                  lumaSigma = 4.0;
                  chromaSigma = 4.0;
              }
          };

          struct bs_params
          {
              float lam;
              float A_diag_min;
              float cg_tol;
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

}

}


#endif //_FastBilateralSolverFilterImpl_HPP_
