
/**@ Test parallel_for and parallel_for_
/**@ Author: chouclee
/**@ 03/17/2013*/
#include <opencv2/opencv.hpp>
#include <time.h>
#include "parallel_for_test.hpp"
using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
	Mat testInput = Mat::ones(40,400000, CV_32F);
	clock_t start, stop;
  // cv::setNumThreads(cv::getNumberOfCPUs());
  cv::setNumThreads(8);

	start = clock();
	parallelTestWithFor(testInput);
	stop = clock();
	cout<<"Running time using \'for\':"<<(double)(stop - start)/CLOCKS_PER_SEC*1000<<"ms"<<endl;

	start = clock();
	parallelTestWithParallel_for(testInput);
	stop = clock();
	cout<<"Running time using \'parallel_for\':"<<(double)(stop - start)/CLOCKS_PER_SEC*1000<<"ms"<<endl;

	start = clock();
	parallelTestWithParallel_for_(testInput);
	stop = clock();
	cout<<"Running time using \'parallel_for_\':"<<(double)(stop - start)/CLOCKS_PER_SEC*1000<<"ms"<<endl;

  cout<<"NumThreads: "<<cv::getNumThreads()<<endl;

	// system("pause");
}
