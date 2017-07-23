cmake_minimum_required(VERSION 3.8)
project(test_nn)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        c_test.c)

add_executable(test_nn ${SOURCE_FILES})