# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

project(faiss
  VERSION 1.7.3
  DESCRIPTION "A library for efficient similarity search and clustering of dense vectors."
  HOMEPAGE_URL "https://github.com/facebookresearch/faiss"
  LANGUAGES CXX)
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic -Wno-psabi -Wno-sign-compare")

IF (NOT OPTIMIZATION_LEVEL)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -mfma")
  add_definitions(-DVAQ_OPTIMIZE)
  add_definitions(-march=haswell)
  add_definitions(-mavx -mavx2)
ELSEIF (OPTIMIZATION_LEVEL STREQUAL "nonavx")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -mfma -mno-avx -mno-avx2")
  add_definitions(-DVAQ_OPTIMIZE)
  add_definitions(-march=haswell)
  # add_definitions(-mavx -mavx2)
ELSEIF (OPTIMIZATION_LEVEL STREQUAL "full")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
ELSEIF (OPTIMIZATION_LEVEL STREQUAL "aggressive")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
ELSEIF (OPTIMIZATION_LEVEL STREQUAL "generic")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
ENDIF()
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# Valid values are "generic", "avx2".
option(FAISS_OPT_LEVEL "" "generic")
option(FAISS_ENABLE_GPU "Enable support for GPU indexes." ON)
option(FAISS_ENABLE_PYTHON "Build Python extension." ON)
option(FAISS_ENABLE_C_API "Build C API." OFF)

if(FAISS_ENABLE_GPU)
  set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
  enable_language(CUDA)
endif()

add_subdirectory(faiss)

if(FAISS_ENABLE_GPU)
  add_subdirectory(faiss/gpu)
endif()
include_directories(external/eigen)
if(FAISS_ENABLE_PYTHON)
  add_subdirectory(faiss/python)
endif()

if(FAISS_ENABLE_C_API)
  add_subdirectory(c_api)
endif()

add_subdirectory(demos)
add_subdirectory(benchs)
add_subdirectory(tutorial/cpp)
add_subdirectory(bitvecengine)
add_subdirectory(examples)
add_definitions(-DEIGEN_USE_BLAS -DEIGEN_USE_LAPACKE)
add_definitions(-DEIGEN_DONT_PARALLELIZE)

FIND_PACKAGE(OpenMP)
IF(OPENMP_FOUND)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
ENDIF()

# CTest must be included in the top level to enable `make test` target.
include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)

  if(FAISS_ENABLE_GPU)
    add_subdirectory(faiss/gpu/test)
  endif()
endif()
