
cmake_minimum_required(VERSION 2.8)
project( Bilateral_solver )


set(CMAKE_BUILD_TYPE "Release")
#set(CMAKE_BUILD_TYPE "Debug")




# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


set(OpenCV_DIR "/home/wangkuan/lib/opencv3/share/OpenCV")
set(Eigen3_DIR "/home/wangkuan/lib/eigen3/share/eigen3/cmake")


find_package(OpenCV REQUIRED )
find_package(Eigen3 REQUIRED)


message( ${OpenCV_DIR})
message( ${Eigen3_DIR})

include_directories(
${OpenCV_INCLUDE_DIRS}
${EIGEN3_INCLUDE_DIR}
)


set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)


add_library(${PROJECT_NAME} SHARED
ICCG.cpp
InputImage.cpp
BilateralGrid.cpp

)




target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
)

add_executable( Colorize colorization.cpp)
target_link_libraries( Colorize ${PROJECT_NAME} )

add_executable( Depthsuperres depthsuperresolution.cpp)
target_link_libraries( Depthsuperres ${PROJECT_NAME} )

add_executable( Latticefilter permutohedral_lattice.cpp)
target_link_libraries( Latticefilter ${PROJECT_NAME} )

add_executable( depth_solver depth.cpp BilateralSolver.hpp)
target_link_libraries( depth_solver ${PROJECT_NAME} )

add_executable( color_solver color.cpp BilateralSolver.hpp)
target_link_libraries( color_solver ${PROJECT_NAME} )

#add_executable( BilateralSolver BilateralSolver.cpp)
#target_link_libraries( BilateralSolver ${PROJECT_NAME} )


#add_executable(SparseMatrix  Sparse-Matrix/src/SparseMatrix/SparseMatrix.cpp Sparse-Matrix/tests/run.cpp)
#target_link_libraries( SparseMatrix ${PROJECT_NAME} )
