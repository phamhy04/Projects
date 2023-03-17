cmake_minimum_required(VERSION 3.1)

project(script)

set(CMAKE_PREFIX_PATH /home/huynhloc/Documents/libtorch)

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}${TORCH_CXX_FLAGS}")

add_executable(${PROJECT_NAME} embed_time.cpp)

target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)