
/**@ Test parallel_for and parallel_for_
/**@ Author: chouclee
/**@ 03/17/2013*/
#include <opencv2/core/internal.hpp>
namespace cv
{
namespace test
{
	class parallelTestBody : public ParallelLoopBody//参考官方给出的answer，构造一个并行的循环体类
	{
	public:
		parallelTestBody(Mat& _src)//class constructor
		{
			src = &_src;
		}
		void operator()(const Range& range) const//重载操作符（）
		{
			Mat& srcMat = *src;
			int stepSrc = (int)(srcMat.step/srcMat.elemSize1());//获取每一行的元素总个数（相当于cols*channels，等同于step1)
			for (int colIdx = range.start; colIdx < range.end; ++colIdx)
			{
				float* pData = (float*)srcMat.col(colIdx).data;
				for (int i = 0; i < srcMat.rows; ++i)
					pData[i*stepSrc] = std::pow(pData[i*stepSrc],3);
			}
		}

	private:
		Mat* src;
	};

	struct parallelTestInvoker//构造一个供parallel_for使用的循环结构体
	{
		parallelTestInvoker(Mat& _src)//struct constructor
		{
			src = &_src;
		}
		void operator()(const BlockedRange& range) const//使用BlockedRange需要包含opencv2/core/internal.hpp
		{
			Mat& srcMat = *src;
			int stepSrc = (int)(srcMat.step/srcMat.elemSize1());
			for (int colIdx = range.begin(); colIdx < range.end(); ++colIdx)
			{
				float* pData = (float*)srcMat.col(colIdx).data;
				for (int i = 0; i < srcMat.rows; ++i)
					pData[i*stepSrc] = std::pow(pData[i*stepSrc],3);
			}
		}
		Mat* src;
	};
}//namesapce test
void parallelTestWithFor(InputArray _src)//'for' loop
{
	CV_Assert(_src.kind() == _InputArray::MAT);
	Mat src = _src.getMat();
	CV_Assert(src.isContinuous());
	int stepSrc = (int)(src.step/src.elemSize1());
	for (int x = 0; x < src.cols; ++x)
	{
		float* pData = (float*)src.col(x).data;
		for (int y = 0; y < src.rows; ++y)
			pData[y*stepSrc] = std::pow(pData[y*stepSrc], 3);
	}
};

void parallelTestWithParallel_for(InputArray _src)//'parallel_for' loop
{
	CV_Assert(_src.kind() == _InputArray::MAT);
	Mat src = _src.getMat();
	int totalCols = src.cols;
	typedef test::parallelTestInvoker parallelTestInvoker;
	parallel_for(BlockedRange(0, totalCols), parallelTestInvoker(src));
};

void parallelTestWithParallel_for_(InputArray _src)//'parallel_for_' loop
{
	CV_Assert(_src.kind() == _InputArray::MAT);
	Mat src = _src.getMat();
	int totalCols = src.cols;
	typedef test::parallelTestBody parallelTestBody;
	parallel_for_(Range(0, totalCols), parallelTestBody(src));
};
}//namespace cv
