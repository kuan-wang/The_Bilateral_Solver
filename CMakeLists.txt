#cmake_minimum_required(VERSION 2.8)
#project( DisplayImage )
#find_package( OpenCV REQUIRED )
#include_directories( ${OpenCV_INCLUDE_DIRS} )
#add_executable( DisplayImage main.cpp ICCG.cpp InputImage.cpp BilateralGrid.cpp)
#target_link_libraries( DisplayImage ${OpenCV_LIBS} )


cmake_minimum_required(VERSION 2.8)
project( Bilateral_solver )



find_package( OpenCV REQUIRED )
#find_package(Eigen3 REQUIRED)

#set(Eigen3_DIR " /usr/local/include/eigen3")

#message( ${Eigen3_DIR})

include_directories(
${OpenCV_INCLUDE_DIRS}
#${EIGEN3_INCLUDE_DIR}
/home/key/eigen3/include/eigen3/
)


set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)


add_library(${PROJECT_NAME} SHARED
ICCG.cpp
InputImage.cpp
BilateralGrid.cpp

)




target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
#${EIGEN3_LIBS}
)

add_executable( Colorize colorization.cpp)
target_link_libraries( Colorize ${PROJECT_NAME} )

add_executable( Depthsuperres depthsuperresolution.cpp)
target_link_libraries( Depthsuperres ${PROJECT_NAME} )

add_executable( Latticefilter permutohedral_lattice.cpp)
target_link_libraries( Latticefilter ${PROJECT_NAME} )


add_executable( BilateralSolver BilateralSolver.cpp)
target_link_libraries( BilateralSolver ${PROJECT_NAME} )


#add_executable(SparseMatrix  Sparse-Matrix/src/SparseMatrix/SparseMatrix.cpp Sparse-Matrix/tests/run.cpp)
#target_link_libraries( SparseMatrix ${PROJECT_NAME} )
