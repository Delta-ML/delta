cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
project(DeltannGraphCompiler)

set(DELTA_INFER_ROOT ${PROJECT_SOURCE_DIR})
set(DELTA_INFER_DELTA_INFER ${DELTA_INFER_ROOT}/delta_infer/)
include_directories(${DELTA_INFER_ROOT})

# compile options
option(BUILD_DEBUG "debug or not" ON)
option(BUILD_SHARED "build shared or static lib" ON)

# platform
option(BUILD_MOBILE "build on android platform" OFF)
option(BUILD_GPU "build on " OFF)
option(BUILD_X86 "build on x86 platform" ON)

# build components
option(BUILD_DELTA_INFER_PYTHON "Build Python binaries." OFF)
option(BUILD_DELTA_INFER_CUSTOM_OPS "Build customed ops lib." OFF)
option(BUILD_WITH_EXAMPLE "Build with example(c++)." OFF)

option(WITH_PYTHON      "Compile with python interpreter"  ON)

# configure environment
set(DELTA_INFER_LINK_LIBS "")
include(cmake/utils.cmake)
include(cmake/build.cmake)
include(cmake/tensorflow.cmake)
include(cmake/external/pybind11.cmake)

if(BUILD_DEBUG)
    add_definitions(-DBUILD_DEBUG)
    set(CMAKE_BUILD_TYPE "Debug")
else()
    add_definitions(-DBUILD_RELEASE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

if(BUILD_MOBILE)
    add_definitions(-DBUILD_MOBILE)
    option(WITH_ARM "" OFF)
    if(WITH_ARM)
        add_definitions(-DWITH_ARM)
    endif()
    option(WITH_VULCAN "" OFF)
    if(WITH_VULCAN)
        add_definitions(-DWITH_VULCAN)
    endif()
endif()

if(BUILD_GPU)
    option(WITH_CUDA "" ON)
    if(WITH_CUDA)
        add_definitions(-DWITH_CUDA)
        include(cmake/cuda.cmake)
    endif()
endif()

if(BUILD_X86)
    add_definitions(-DBUILD_X86)
    option(WITH_OMP "" ON)
    if(WITH_OMP) 
        add_definitions(-DWITH_OMP)
        find_openmp()
    endif()
endif()

add_subdirectory(${DELTA_INFER_DELTA_INFER})

if(BUILD_DELTA_INFER_PYTHON)
    add_subdirectory(${DELTA_INFER_ROOT}/python)
endif()

if(BUILD_WITH_EXAMPLE)
    add_subdirectory(${DELTA_INFER_ROOT}/example/c++/)
endif()
