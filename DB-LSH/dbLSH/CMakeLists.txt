cmake_minimum_required(VERSION 3.0)
project(dblsh)

# Set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11 -Wall -ffast-math -march=native")

find_package(Boost REQUIRED)
if (Boost_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${Boost_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Boost_CXX_FLAGS}")
    include_directories(${Boost_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "Boost dynamic-bitset is required")
endif()

# Find all source files
file(GLOB SRCS src/*.cpp)

# Add executable target
add_executable(dblsh ${SRCS})

# Define clean targets
add_custom_target(clean-all
    COMMAND ${CMAKE_COMMAND} -P clean-all.cmake
)
add_custom_target(clean-txt
    COMMAND rm -rf ${CMAKE_BINARY_DIR}/*.txt
)
add_custom_target(clean-codes
    COMMAND rm -rf ${CMAKE_SOURCE_DIR}/*.cpp ${CMAKE_SOURCE_DIR}/*.h ${CMAKE_SOURCE_DIR}/src/*.cpp ${CMAKE_SOURCE_DIR}/src/*.p
)

# Generate clean-all.cmake
file(WRITE ${CMAKE_BINARY_DIR}/clean-all.cmake "
    if(EXISTS \"${CMAKE_BINARY_DIR}/CMakeCache.txt\")
        file(REMOVE \"${CMAKE_BINARY_DIR}/CMakeCache.txt\")
    endif()
    if(EXISTS \"${CMAKE_BINARY_DIR}/Makefile\")
        execute_process(COMMAND make clean WORKING_DIRECTORY \"${CMAKE_BINARY_DIR}\")
    endif()
")
