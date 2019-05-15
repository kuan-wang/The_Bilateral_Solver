
# The Fast Bilateral Solver (**Contributed in OpenCV**)

**The Bilater Solver** is a novel algorithm for edge-aware smoothing that combines the flexibility and speed of simple filtering approaches with the accuracy of domain-specific optimization algorithms. This algorithm was presented by Jonathan T. Barron and Ben Poole as an ECCV2016 oral and best paper nominee. Algorithm details and applications can be found in https://arxiv.org/pdf/1511.03296.pdf.
- [**New!**] This repo is the draft codes for my [GSOC2017](https://summerofcode.withgoogle.com/archive/2017/projects/6379759588081664/) (google summer of code) project. The [opencv_contrib](https://github.com/kuan-wang/opencv_contrib) branch I [pull request](https://github.com/opencv/opencv_contrib/pull/1317) to OpenCV was not successfully [merged](https://github.com/opencv/opencv_contrib/pull/1819) in master after GSOC2017 and I've no time to fix the issues. 
- Fortunately and thankfully, last year this project has been contributed to opencv as a file [fbs_filter.cpp](https://github.com/opencv/opencv_contrib/blob/master/modules/ximgproc/src/fbs_filter.cpp) in ximgproc of opencv_contrib with the [help](https://github.com/opencv/opencv_contrib/pull/1819) of [berak](https://github.com/berak), [Alexander Alekhin](https://github.com/alalek) and [Jukka Komulainen](https://github.com/ytyytyyt). Really thanks!
- If you want to use the cpp version of fast bilateral solver, just compile the opencv with opencv_contrib. And This repo has some interesting demos you can play.

## Introduce
### Algorithm

![SBS](https://raw.githubusercontent.com/THUKey/The_Bilateral_Solver/4cff9dabc9ad48d047f66cc8d68c733a1e403688/build/SBS.png)
- **Splat+Blur+Slice Procedure** 
The two bilateral representations we use in this project, here shown filtering a toy one-dimensional grayscale image of a step-edge. This toy image corresponds to a 2D space visualized here (x = pixel location, y = pixel value) while in the paper we use RGB images, which corresponds to a 5D space (XYRGB). The lattice (Fig 2a) uses barycen-tric interpolation to map pixels to vertices and requires d+1 blurring operations, where d is the dimensionality of the space. The simplified bilateral grid (Fig 2b) uses nearest-neighbor interpolation and requires d blurring operations which are summed rather than done in sequence. The grid is cheaper to construct and to use than the lattice, but the use of hard assignments means that the filtered output often has blocky piecewise-constant artifacts.

- **Diagrammatize**
```flow
st=>start: Start
e=>end

inr=>operation: Imput reference image
int=>operation: Imput target image
bg=>operation: construct BilateralGrid
sl=>operation: construct SliceMatrix
bl=>operation: construct BlurMatrix
A1=>operation: construct AMatrix step1
A2=>operation: construct AMatrix step2
cg=>operation: execute ICCG
out=>operation: output the resolt


st->inr->bg->sl->bl->A1->int->A2->cg->out->e


```


__________
## Installation Instructions
### Build OpenCV
This is just a suggestion on how to build OpenCV 3.1. There a plenty of options. Also some packages might be optional.
```
sudo apt-get install libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev
git clone https://github.com/Itseez/opencv.git
cd opencv
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE -D WITH_CUDA=OFF ..
make -j
sudo make install
```

### Build The_Bilateral_Solver
```
git clone https://github.com/THUKey/The_Bilateral_Solver.git
cd The_Bilateral_Solver/build
cmake ..
make
```
This will create three executable demos, that you can run as shown in below.

#### Depthsuperresolution

![target](https://raw.githubusercontent.com/THUKey/The_Bilateral_Solver/master/build/target.png)
the target.
```
./Depthsuperres
```
![result](https://raw.githubusercontent.com/THUKey/The_Bilateral_Solver/master/build/depthsuperresolution.png)
This result(use bilateral solver) is far from the optimal performance, which means there are some extra work to do, such as to patiently adjustment parameters and to optimize the implementation.
```
./Latticefilter reference.png target.png
```
 ![enter image description here](https://raw.githubusercontent.com/THUKey/The_Bilateral_Solver/master/build/lattice_output.png)
 This result(use permutohedral_lattice) is quite nice.
#### Colorization
```
./Colorize rose1.webp
```
![draw](https://raw.githubusercontent.com/THUKey/The_Bilateral_Solver/master/build/draw.png)
draw image, then press "ESC" twice to launch the colorization procession.
![colorized](https://raw.githubusercontent.com/THUKey/The_Bilateral_Solver/master/build/colorized.png)
colorized image.
you could change the **rose1.webp** to your own image. Thanks for [timuda](https://github.com/timuda/colorization_s_demo), his colorization implementation help me a lot.

#### PermutohedralLatticeFilter
```
./Latticefilter flower8.jpg
```
In Barron's another paper *Fast Bilateral-Space Stereo for Synthetic Defocus*, both bileteral_solver and permutohedral lattice are used to do experiment, and the result shows that bilateral_solver is  faster than permutohedral lattice technique, but the permutohedral is more accurate than the bilateral_solver. In other words, this is the tradeoff between time and accuracy. Actually, both two techniques' tradeoff can be worthwhile in appropriate condition. So I want to implement both two technique for more widely use.
![output](https://raw.githubusercontent.com/THUKey/The_Bilateral_Solver/master/build/lattice_flower8.png)
filter_output.
![input](https://raw.githubusercontent.com/THUKey/The_Bilateral_Solver/master/build/flower8.jpg)
filter_input.


__________
## Basic Usage
### Depthsuperresolution:
```
	BilateralGrid BiGr(mat_R);
	BiGr.Depthsuperresolution(mat_R,mat_T,sigma_spatial,sigma_luma,sigma_chroma);
```
Firstly, we use the reference image mat_R construct a BilateralGrid, the we launch a depthsuperresolution to optimize the target image mat_T. The parameter sigma_spatial is the Gaussian kernal for coordinate x y, similarly , the sigma_luma correspond luma(Y) and the sigma_chroma correspond chroma(UV). It need to be noted that he mat_R should be covert to YUV form before construct the bilateralgrid.

### Colorization
```
	InputImage InImg(mat_in);
	mat_bg_in = InImg.get_Image(IMG_YUV);
	InImg.draw_Image();
	mat_bg_draw_in = InImg.get_Image(IMG_DRAWYUV);
	BilateralGrid BiGr(mat_bg_in);
	BiGr.Colorization(mat_in,mat_bg_draw_in);

```
Similar to above, we need to covert the imput image mat_in(gray image for colorization) to YUV form, then draw the gray image. when the drawing finished, press "ESC" twice to launch the colorization procession. the result will be save in specified folder.
### PermutohedralLattce
```
	bilateral(im,spatialSigma,colorSigma);
```
Similar to BilateralGrid, the PermutohedralLattce also need spatial parameter and the color parameter to specified the Gaussian kernel.


__________
## Schedule
| Item      |   State  |   Remark|
| :-------- | --------:| :--: |
|C++ code of the core algorithm   | Completed | also python   |
|Depthsuperres module |   Completed |  need optimize  |
|Colorization module |   Completed |choose ICCG or others|
|PermutohedralLatticeFilter   | Completed |increse Compatibility |
|Semantic Segmentation optimizer |   Ongoing |  try apply in CNN
|Contribute project to OpenCV   |    Ongoing | coding testfile  |
|Detail Documentation  | Ongoing | writing toturial   |


----
## Reference
```
article{BarronPoole2016,
author = {Jonathan T Barron and Ben Poole},
title = {The Fast Bilateral Solver},
journal = {ECCV},
year = {2016},
}
@article{Barron2015A,
author = {Jonathan T Barron and Andrew Adams and YiChang Shih and Carlos Hern\'andez},
title = {Fast Bilateral-Space Stereo for Synthetic Defocus},
journal = {CVPR},
year = {2015},
}
@article{Adams2010,
author = {Andrew Adams	Jongmin Baek	Abe Davis},
title = {Fast High-Dimensional Filtering Using the Permutohedral Lattice},
journal = {Eurographics},
year = {2010},
}
```
