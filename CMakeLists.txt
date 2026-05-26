cmake_minimum_required(VERSION 3.14)
project(DGTD_MOR_Testing LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


include(FetchContent)
FetchContent_Declare(
  Eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
)
FetchContent_MakeAvailable(Eigen)

add_library(dgtd_mor STATIC
    src/MFEMHandler.cpp
    src/POD_Basis.cpp
    src/ModelOrderReduction.cpp
)
target_include_directories(dgtd_mor PUBLIC src)
target_link_libraries(dgtd_mor PUBLIC Eigen3::Eigen)

