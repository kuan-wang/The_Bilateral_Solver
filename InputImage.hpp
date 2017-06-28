#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

#define MARK_RADIUS (10)
using namespace cv;
using namespace std;

enum ShowInImg
{
	IMG_INPUT = 0,
	IMG_GRAY,
	IMG_DRAW,
	IMG_YUV,
	IMG_DRAWYUV,
	IMG_OUTPUT,
	IMG_COLORIZED,
	IMG_NUM
};

static void my_mouse_callback(int event, int x, int y, int flags, void* param);
class InputImage
{
	public:
		InputImage(Mat1f mat_image);
		InputImage(Mat3f mat_image);
		Mat3f get_Image(int num);
		void draw_Image(void);			
		void show_Image(int num);

	private:
		Mat3f copy_GlaychForRGBch(Mat1f, Mat3f);
		void draw_Trajectory(Mat3f *);
		Mat3f mat_input;
		Mat3f mat_draw;
		Mat3f mat_draw_bp;
		Mat3f mat_draw_yuv;
		Mat3f mat_yuv;
		Mat1f mat_gray;
		bool mouse_click;
		bool mouse_left;
		int mouse_x;
		int mouse_y;
};
