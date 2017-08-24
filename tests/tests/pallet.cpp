//====================================================================      
// 作者   : lishichao
// 日期   : 2014年05月04日      
// 描述   : HSV颜色盘     
//====================================================================  
//#include "stdafx.h" 
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#define max(a,b)  (((a)>(b))?(a):(b))  
#define min(a,b)  (((a)<(b))?(a):(b))  
using namespace cv;

double module(Point pt)
{
	return sqrt((double)pt.x*pt.x + pt.y*pt.y);
}

double distance(Point pt1, Point pt2)
{
	int dx = pt1.x - pt2.x;
	int dy = pt1.y - pt2.y;
	return sqrt((double)dx*dx + dy*dy);
}

double cross(Point pt1, Point pt2)
{
	return pt1.x*pt2.x + pt1.y*pt2.y;
}

double angle(Point pt1, Point pt2)
{
	return acos(cross(pt1, pt2) / (module(pt1)*module(pt2) + DBL_EPSILON));
}
// p和c其中一个是圆心  
int inCircle(Point p, Point c, int r)
{
	int dx = p.x - c.x;
	int dy = p.y - c.y;
	return dx*dx + dy*dy <= r*r ? 1 : 0;

}

//画出hsv圆盘
void createPlate(Mat &im1, int radius)
{
	Mat hsvImag(Size(radius << 1, radius << 1), CV_8UC3, Scalar(0, 0, 255));
	int w = hsvImag.cols;
	int h = hsvImag.rows;
	int cx = w >> 1;
	int cy = h >> 1;
	Point pt1(cx, 0);

	for (int j = 0; j < w; j++)
	{
		//uchar* data = hsvImag.ptr<uchar>(j);

		for (int i = 0; i < h; i++)
		{
			Point pt2(j - cx, i - cy);
			if (inCircle(Point(0, 0), pt2, radius))
			{
				int theta = angle(pt1, pt2) * 180 / CV_PI;
				if (i > cx)
				{
					theta = -theta + 360;
				}
				//cout << theta << ' ' << module(pt2) / cx * 255 << endl;
				//data[i * 3 + 0] = theta / 2;
				//////data[i * 3 + 1] = module(pt2) / cx * 255;
				//data[i * 3 + 2] = 255;
				hsvImag.at<Vec3b>(i, j)[0] = theta / 2;
				hsvImag.at<Vec3b>(i, j)[1] = module(pt2) / cx * 255;
				hsvImag.at<Vec3b>(i, j)[2] = 255;
			}
		}
	}

	
	cvtColor(hsvImag, im1, CV_HSV2BGR);
	//im1 =  hsvImag;
}
int main()
{
	Mat img;
	int radius = 255;
	createPlate(img, radius);

	namedWindow("img");
	imshow("img", img);
	waitKey(0);
}
