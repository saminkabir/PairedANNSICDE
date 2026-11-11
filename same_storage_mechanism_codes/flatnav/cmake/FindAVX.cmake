cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

# FLATNAV_ENABLE_AVX512=OFF (default) builds the AVX2 kernels even on CPUs that
# support AVX-512; set it to ON to restore the previous auto-detected AVX-512.
option(FLATNAV_ENABLE_AVX512 "Compile FlatNav with AVX-512 when the CPU supports it" OFF)

if(CMAKE_SYSTEM_NAME MATCHES "Linux")
  execute_process(COMMAND cat /proc/cpuinfo OUTPUT_VARIABLE CPUINFO OUTPUT_STRIP_TRAILING_WHITESPACE)

  string(FIND "${CPUINFO}" "sse" SSE_FOUND)
  string(FIND "${CPUINFO}" "sse3" SSE3_FOUND)
  string(FIND "${CPUINFO}" "avx" AVX_FOUND)
  string(FIND "${CPUINFO}" "avx2" AVX2_FOUND)
  string(FIND "${CPUINFO}" "avx512" AVX512_FOUND)

  if(SSE_FOUND GREATER -1)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse")
    message(STATUS "Building with SSE")
  endif()

  if(SSE3_FOUND GREATER -1)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse3")
    message(STATUS "Building with SSE3")
  endif()

  if(AVX_FOUND GREATER -1)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
    message(STATUS "Building with AVX")
  endif()

  if(AVX2_FOUND GREATER -1)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
    message(STATUS "Building with AVX2")
  endif()

  if(AVX512_FOUND GREATER -1 AND FLATNAV_ENABLE_AVX512)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f -mavx512dq -mavx512vl -mavx512bw -mavx512vnni")
    message(STATUS "Building with AVX512")
  elseif(AVX512_FOUND GREATER -1)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mno-avx512f")
    message(STATUS "AVX512 available but disabled (FLATNAV_ENABLE_AVX512=OFF): using AVX2")
  endif()

elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
  execute_process(COMMAND /usr/sbin/sysctl -n machdep.cpu.features OUTPUT_VARIABLE CPUINFO OUTPUT_STRIP_TRAILING_WHITESPACE)

  string(FIND "${CPUINFO}" "SSE" SSE_FOUND)
  string(FIND "${CPUINFO}" "SSE3" SSE3_FOUND)

  if(SSE_FOUND GREATER -1)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse")
    message(STATUS "Building with SSE")
  endif()

  if(SSE3_FOUND GREATER -1)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse3")
    message(STATUS "Building with SSE3")
  endif()
endif()
