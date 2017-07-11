
cmake_minimum_required(VERSION 2.8)
project( Bilateral_solver )

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



find_package( OpenCV REQUIRED )
#find_package(Eigen3 REQUIRED)

#set(Eigen3_DIR " /usr/local/include/eigen3")

#message( ${Eigen3_DIR})

include_directories(
${OpenCV_INCLUDE_DIRS}
#${EIGEN3_INCLUDE_DIR}
/home/key/eigen3/include/eigen3/
./tests/
)


#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)


add_library(${PROJECT_NAME} SHARED
run.cpp

)




target_link_libraries(
${PROJECT_NAME}
${OpenCV_LIBS}
#${EIGEN3_LIBS}
)

add_executable(Test run.cpp)
target_link_libraries(Test ${PROJECT_NAME} )
