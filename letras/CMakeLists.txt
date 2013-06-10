cmake_minimum_required(VERSION 2.8)

find_package( OpenCV REQUIRED )
include_directories("${CMAKE_SOURCE_DIR}")

add_executable( letters letters.cpp )
target_link_libraries( letters ${OpenCV_LIBS} )
