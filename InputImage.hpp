#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

#define MARK_RADIUS (10)
using namespace cv;
using namespace std;

//途中経過確認用の変数
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
		InputImage(Mat1f mat_image);	//コンストラクた
		Mat3f get_Image(int num);		//yuv画像のGetter
		void draw_Image(void);			//ユーザのカラー指定
		void show_Image(int num);		//デバック用の画像表示

	private:
		Mat3f copy_GlaychForRGBch(Mat1f, Mat3f);	//グレー1ch画像を3chに拡張する
		void draw_Trajectory(Mat3f *);				//マウス入力で軌跡を描く
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
