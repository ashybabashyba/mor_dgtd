cmake_minimum_required(VERSION 3.14)
project(DGTD_MOR_Testing LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

file(GLOB_RECURSE MOR_SOURCES "src/mor/*.cpp")
file(GLOB_RECURSE DRIVER_SOURCES "src/driver/*.cpp")
file(GLOB_RECURSE LAUNCHER_SOURCES "src/launcher/*.cpp")

set(CORE_SOURCES ${MOR_SOURCES} ${DRIVER_SOURCES} ${LAUNCHER_SOURCES})
list(REMOVE_ITEM CORE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/launcher/launcher.cpp")

add_library(mor_core STATIC ${CORE_SOURCES})
target_include_directories(mor_core PUBLIC 
    src/mor 
    src/driver 
    src/launcher
)
target_link_libraries(mor_core PRIVATE 
    Eigen3::Eigen 
    nlohmann_json::nlohmann_json
)

add_executable(mor_solver src/launcher/launcher.cpp)
target_link_libraries(mor_solver PRIVATE mor_core)

enable_testing()
add_subdirectory(test)