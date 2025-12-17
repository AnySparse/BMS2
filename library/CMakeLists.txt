

cmake_minimum_required(VERSION 3.10)

project(SIGMO LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


include(${CMAKE_SOURCE_DIR}/cmake/SetupSYCL.cmake)
find_package(MPI REQUIRED)


include_directories(${CMAKE_SOURCE_DIR}/include)


set(CXXOPTS_LOCAL_DEFAULT "/cxxopts-master/cxxopts-master/include")
if(NOT DEFINED CXXOPTS_LOCAL AND EXISTS "${CXXOPTS_LOCAL_DEFAULT}/cxxopts.hpp")
  set(CXXOPTS_LOCAL "${CXXOPTS_LOCAL_DEFAULT}")
endif()

if(CXXOPTS_LOCAL AND EXISTS "${CXXOPTS_LOCAL}/cxxopts.hpp")
  message(STATUS "Using local cxxopts at: ${CXXOPTS_LOCAL}")
  add_library(cxxopts_headers INTERFACE)
  target_include_directories(cxxopts_headers INTERFACE "${CXXOPTS_LOCAL}")
else()
  message(STATUS "Local cxxopts not found. Falling back to FetchContent.")
  include(FetchContent)
  FetchContent_Declare(
    cxxopts
    GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
    GIT_TAG v2.2.1
  )
  FetchContent_MakeAvailable(cxxopts)
  add_library(cxxopts_headers INTERFACE)
  target_include_directories(cxxopts_headers INTERFACE "${cxxopts_SOURCE_DIR}/include")
endif()


set(SIGMO_ALLOCATION "device" CACHE STRING "Location of the graph: host, device, or shared")
set_property(CACHE SIGMO_ALLOCATION PROPERTY STRINGS "host" "device" "shared")
if (SIGMO_ALLOCATION STREQUAL "host")
  message(WARNING "GRAPH_LOCATION=host is not recommended for performance. Use at your own risk.")
  add_compile_definitions(SIGMO_ALLOCATION=0)
elseif (SIGMO_ALLOCATION STREQUAL "device")
  add_compile_definitions(SIGMO_ALLOCATION=1)
elseif (SIGMO_ALLOCATION STREQUAL "shared")
  add_compile_definitions(SIGMO_ALLOCATION=2)
else()
  message(FATAL_ERROR "Invalid SIGMO_ALLOCATION value. Must be 'host', 'device', or 'shared'.")
endif()


option(SIGMO_ENABLE_TEST "Enable tests" OFF)
if (SIGMO_ENABLE_TEST)
  add_subdirectory(${CMAKE_SOURCE_DIR}/test)
endif()


add_executable(BMS2
  ${CMAKE_SOURCE_DIR}/src/BMS2.cpp
)
add_executable(BMS2_mpi
  ${CMAKE_SOURCE_DIR}/src/BMS2_mpi.cpp
)
add_executable(BMS2_batch
  ${CMAKE_SOURCE_DIR}/src/BMS2_batch.cpp
)

target_link_libraries(BMS2     PRIVATE cxxopts_headers)
target_link_libraries(BMS2_batch    PRIVATE cxxopts_headers)
target_link_libraries(BMS2_mpi PRIVATE MPI::MPI_CXX cxxopts_headers) 

set(SIGMO_TARGET_ARCHITECTURE "" CACHE STRING "Target architecture for the SIGMO library")
if (SIGMO_TARGET_ARCHITECTURE STREQUAL "")
  message(FATAL_ERROR "Please specify -DSIGMO_TARGET_ARCHITECTURE, e.g. nvptx64-nvidia-cuda or spir64.")
endif()


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=${SIGMO_TARGET_ARCHITECTURE} -g -fno-sycl-id-queries-fit-in-int")
