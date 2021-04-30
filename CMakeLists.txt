cmake_minimum_required(VERSION 3.5.1 FATAL_ERROR)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")
set(CMAKE_BUILD_TYPE "Debug")

set(THIRDPARTY ../../3rdparty)

# 使CUDA NVCC 编译器支持C++11特性
LIST(APPEND CUDA_NVCC_FLAGS -std=c++11;-O2)
LIST(APPEND CUDA_NVCC_FLAGS -Xcompiler;-fPIC)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_50,code=sm_50)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_52,code=sm_52)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_60,code=sm_60)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_61,code=sm_61)

#opencv
set(OPENCV_VERSION 348)
set(OPENCV_INLCUDE_DIR ${THIRDPARTY}/opencv/linux/include)
set(OPENCV_LIB_DIR ${THIRDPARTY}/opencv/linux/lib)
set(OPENCV_LIBS opencv_core opencv_highgui opencv_imgproc opencv_videoio opencv_imgcodecs opencv_features2d opencv_xfeatures2d opencv_calib3d opencv_flann)
#opencv_imgcodecs
include_directories(${OPENCV_INLCUDE_DIR})
link_directories(${OPENCV_LIB_DIR})

#cuda
find_package(CUDA REQUIRED)
message(STATUS "CUDA library status:")
message(STATUS "    version: ${CUDA_VERSION}")
message(STATUS "    libraries: ${CUDA_LIBRARIES}")
message(STATUS "    libraries: ${CUDA_CUDA_LIBRARY}")
message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")
INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(/usr/local/cuda/targets/x86_64-linux/include)
#link_directories(/usr/local/cuda/targets/x86_64-linux/lib)
link_directories(/usr/local/cuda/lib64)

#tensorrt
set(tensorRT_INCLUDE_DIR /media/zw/DL/ly/software/TensorRT-7.1.3.4/include)
set(tensorRT_LIB_DIR /media/zw/DL/ly/software/TensorRT-7.1.3.4/lib)
set(tensorRT_LIBS nvinfer nvparsers nvinfer_plugin nvonnxparser)
include_directories(${tensorRT_INCLUDE_DIR})
link_directories(${tensorRT_LIB_DIR})
include_directories(/media/zw/DL/ly/software/TensorRT-7.1.3.4/samples/common)
set(logger_file "/media/zw/DL/ly/software/TensorRT-7.1.3.4/samples/common/logger.cpp")

# cudnn
set(CUDNN_LIB_DIR ${THIRDPARTY}/cudnn/lib64)
set(CUDNN_LIBS cudnn)
link_directories(${CUDNN_LIB_DIR})


#aux_source_directory(. SRC_LIST)
add_library(AlgCrowdCount STATIC crowdcount.cpp crowdcount.h ${logger_file})
target_link_libraries(AlgCrowdCount ${tensorRT_LIBS} ${CUDNN_LIBS} ${OPENCV_LIBS} ${CUDA_nvcuvid_LIBRARY} ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY} cudart)

add_executable(CROWD test.cpp crowdcount.cpp crowdcount.h ${logger_file})
target_link_libraries(CROWD ${tensorRT_LIBS} ${CUDNN_LIBS} ${OPENCV_LIBS} ${CUDA_nvcuvid_LIBRARY} ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY} cudart)


