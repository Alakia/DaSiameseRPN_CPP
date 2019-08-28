cmake_minimum_required (VERSION 2.8)

SET(OpenCV_DIR "/usr/local/opencv/opencv3.4/share/OpenCV")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -march=native -std=c++11")

find_package(OpenCV REQUIRED)

LINK_DIRECTORIES("${PROJECT_SOURCE_DIR}/build/lib")

SET(Torch_DIR "/home/marco/Software/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

include_directories(.
        ${PROJECT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
        "/usr/local/include/eigen3"
        ${TORCH_INCLUDE_DIRS}
        )

ADD_EXECUTABLE(demo
        src/demo.cpp
        src/SiamTracker.cpp
        src/Exception.cpp
        src/SiameseRPN.cpp
        include/Exception.h
        include/Interface.h
        include/Processor.h
        include/SiamTracker.h
        include/SiameseRPN.h
        include/Storage.h
        include/utils.h
        )

TARGET_LINK_LIBRARIES(demo
        ${OpenCV_LIBS}
        ${TORCH_LIBRARIES}
        )