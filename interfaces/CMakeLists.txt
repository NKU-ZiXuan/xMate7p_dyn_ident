project(cPlusPlusCourses)

cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAG} -std=c++11 -Wall")

include_directories(
    include
    /usr/include/eigen3
)

add_executable(hw1 src/hw1.cpp)
add_executable(hw2 src/hw2.cpp)
add_executable(hw3 src/hw3.cpp)
add_executable(hw4 src/hw4.cpp)
add_executable(hw5 src/hw5.cpp)
add_executable(hw6 src/hw6.cpp)
add_executable(hw7 src/hw7.cpp)
add_executable(hw8 src/hw8.cpp)
add_executable(hw.6.2.1 src/hw.6.2.1.cpp )
add_executable(hw.6.2.5 src/hw.6.2.5.cpp )
add_executable(test src/test.cpp )

add_executable(compute_torque src/compute_torque.cpp )
