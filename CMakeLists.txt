cmake_minimum_required(VERSION 3.20)
project(InstanceSegment)

set(CMAKE_CXX_STANDARD 17)


find_package(OpenCV 3.4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


#Libtorch
set(CMAKE_PREFIX_PATH "/usr/local/lib;/home/chen/app/libtorch")

find_package(TorchVision REQUIRED)
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})

#CUDA
find_package(CUDA 10.2 REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

set(TensorRT_LIBS nvinfer nvonnxparser nvinfer_plugin)
include_directories(${TensorRT_INCLUDE_DIRS})


aux_source_directory(InstanceSegment InstanceSegmentation_SOURCES)

add_executable(InstanceSegment main.cpp ${InstanceSegmentation_SOURCES})
target_link_libraries(InstanceSegment  ${CUDA_LIBRARIES} ${OpenCV_LIBRARIES}  ${TORCH_LIBRARIES}
        TorchVision::TorchVision ${TensorRT_LIBS})

add_executable(demo demo.cpp ${InstanceSegmentation_SOURCES})
target_link_libraries(demo  ${CUDA_LIBRARIES} ${OpenCV_LIBRARIES}  ${TORCH_LIBRARIES}
        TorchVision::TorchVision ${TensorRT_LIBS})

add_executable(build_model build_model.cpp InstanceSegment/parameters.cpp InstanceSegment/logger.cpp)
target_link_libraries(build_model  ${TensorRT_LIBS} pthread ${OpenCV_LIBRARIES})

