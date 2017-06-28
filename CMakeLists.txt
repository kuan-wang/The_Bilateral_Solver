#cmake_minimum_required(VERSION 2.8)
#project( DisplayImage )
#find_package( OpenCV REQUIRED )
#include_directories( ${OpenCV_INCLUDE_DIRS} )
#add_executable( DisplayImage main.cpp ICCG.cpp InputImage.cpp BilateralGrid.cpp)
#target_link_libraries( DisplayImage ${OpenCV_LIBS} )


cmake_minimum_required(VERSION 2.8)
project( Bilateral_solver )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )


add_executable( Colorize colorization.cpp ICCG.cpp InputImage.cpp BilateralGrid.cpp)
target_link_libraries( Colorize ${OpenCV_LIBS} )

add_executable( Depthsuperres depthsuperresolution.cpp ICCG.cpp InputImage.cpp BilateralGrid.cpp)
target_link_libraries( Depthsuperres ${OpenCV_LIBS} )

add_executable( Latticefilter permutohedral_lattice.cpp)
target_link_libraries( Latticefilter ${OpenCV_LIBS} )
