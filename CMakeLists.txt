cmake_minimum_required(VERSION 3.1)
project(MyLDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Wextra -Wpedantic")

add_executable(lda LDA.cpp)
add_executable(plsa PLSA.cpp)

