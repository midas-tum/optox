# @project optox - operator to X 
# @author Erich Kobler <erich.kobler@icg.tugraz.at>
# @date 01.07.2018

project(optoX)

cmake_minimum_required(VERSION 3.12)

##----------------------------------------------------------------------------
## CMake definitions
##----------------------------------------------------------------------------
set(CMAKE_PROPAGATE_HOST_FLAGS OFF)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)

##-----------------------------------------------------------------------------
## Library output directory
##-----------------------------------------------------------------------------
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
#set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY TRUE) 

##-----------------------------------------------------------------------------
# CUDA
##-----------------------------------------------------------------------------
find_package(CUDA 5.0 REQUIRED)
cuda_include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
if(NOT DEFINED $ENV{CUDA_SDK_ROOT_DIR})
  message(STATUS, "cuda sdk not set")
  set(CUDA_SDK_ROOT_DIR ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc)
else()
  set(CUDA_SDK_ROOT_DIR $ENV{CUDA_SDK_ROOT_DIR}/common/inc)
endif(NOT DEFINED $ENV{CUDA_SDK_ROOT_DIR})
cuda_include_directories(${CUDA_SDK_ROOT_DIR})
include_directories(${CUDA_SDK_ROOT_DIR})

message(STATUS "CUDA_TOOLKIT_ROOT_DIR: ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "CUDA_SDK_ROOT_DIR: ${CUDA_SDK_ROOT_DIR}")

##-----------------------------------------------------------------------------
# Compiler flags
##-----------------------------------------------------------------------------
include(${CMAKE_CURRENT_LIST_DIR}/settings.cmake)
set(ENV{LDFLAGS} -L${CUDA_TOOLKIT_ROOT_DIR}/lib64)
message(STATUS "LDFLAGS = $ENV{LDFLAGS}")

##-----------------------------------------------------------------------------
## gpuNUFFT
##-----------------------------------------------------------------------------
option(WITH_GPUNUFFT "Use gpuNUFFT for non-uniform MRI operator. Requires GPUNUFFT_ROOT to be set." OFF)
message(STATUS "WITH_GPUNUFFT... ${WITH_GPUNUFFT}")

if(WITH_GPUNUFFT)
    message(STATUS "Use gpuNUFFT (required for non-uniform MRI operator)")
    find_package(GPUNUFFT REQUIRED)
    include_directories(${GPUNUFFT_INCLUDE_DIRS})
else(WITH_GPUNUFFT)
    message(STATUS "Disable gpuNUFFT (required for non-uniform MRI operator)")
endif(WITH_GPUNUFFT)

##-----------------------------------------------------------------------------
## sources
##-----------------------------------------------------------------------------
include_directories(src)
add_subdirectory(src)

##-----------------------------------------------------------------------------
## optox include and library directories
##-----------------------------------------------------------------------------
set(OPTOX_LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(OPTOX_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
message(STATUS "optox include dir: ${OPTOX_INCLUDE_DIR}")
message(STATUS "optox library dir: ${OPTOX_LIBRARY_DIR}")

##-----------------------------------------------------------------------------
## interfaces
##-----------------------------------------------------------------------------
option(WITH_PYTHON "Use Python - build python interface. Requires numpy, pybind11" ON)
if(WITH_PYTHON)
    message(STATUS "[+] optox build python interfaces")
    add_subdirectory(python)
else(WITH_PYTHON)
    message(STATUS "[-] optox exclude python interfaces")
endif(WITH_PYTHON)

option(WITH_TENSORFLOW "Use Tensorflow - build tensorflow interface. Requires tensorflow" ON)
if(WITH_TENSORFLOW)
    message(STATUS "[+] optox build tensorflow interfaces")
    add_subdirectory(tensorflow)
else(WITH_TENSORFLOW)
    message(STATUS "[-] optox exclude tensorflow interfaces")
endif(WITH_TENSORFLOW)

option(WITH_PYTORCH "Use Pytorch - build pytorch interface. Requires pytorch" ON)
if(WITH_PYTORCH)
    message(STATUS "[+] optox build pytorch interfaces")
    add_subdirectory(pytorch)
else(WITH_PYTORCH)
    message(STATUS "[-] optox exclude pytorch interfaces")
endif(WITH_PYTORCH)

enable_testing()

# look for Doxygen package
find_package(Doxygen)
if (DOXYGEN_FOUND)
    # set input and output files
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}//Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile.out)

    # request to configure the file
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    message("Doxygen build started")

    # Note: do not put "ALL" - this builds docs together with application EVERY TIME!
    add_custom_target( docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM )
else (DOXYGEN_FOUND)
  message("Doxygen needs to be installed to generate the doxygen documentation")
endif (DOXYGEN_FOUND)
