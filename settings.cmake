
  set(MY_NVCC_FLAGS -arch=sm_50 -gencode arch=compute_50,code=sm_50)
  list(APPEND MY_NVCC_FLAGS -gencode=arch=compute_52,code=sm_52)
  list(APPEND MY_NVCC_FLAGS -gencode=arch=compute_52,code=compute_52)  

  if(${CUDA_VERSION_MAJOR} GREATER 7)
    message("add CUDA 8 flags: Current version: ${CUDA_VERSION}")
    list(APPEND MY_NVCC_FLAGS -gencode=arch=compute_60,code=sm_60)
    list(APPEND MY_NVCC_FLAGS -gencode=arch=compute_61,code=sm_61)
    list(APPEND MY_NVCC_FLAGS -gencode=arch=compute_61,code=compute_61)
  endif()

  if(${CUDA_VERSION_MAJOR} GREATER 8)
    message("add CUDA 9 flags: Current version: ${CUDA_VERSION}")
    list(APPEND MY_NVCC_FLAGS -gencode=arch=compute_70,code=sm_70)
  endif()

  if(${CUDA_VERSION_MAJOR} GREATER 9)
    message("add CUDA 10 flags: Current version: ${CUDA_VERSION}")
    list(APPEND MY_NVCC_FLAGS -gencode=arch=compute_75,code=sm_75)
  endif()

  if(${CUDA_VERSION_MAJOR} GREATER 10)
    message("add CUDA 11 flags: Current version: ${CUDA_VERSION}")
    list(APPEND MY_NVCC_FLAGS -gencode=arch=compute_86,code=sm_86)
  endif()
set(CUDA_NVCC_FLAGS ${MY_NVCC_FLAGS})

#------------- for further compiler flags see compiler_settings.cmake.txt -----------------
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()
include(${CMAKE_CURRENT_LIST_DIR}/cmake/compiler_settings.cmake)
#----------------------------------------------------------------------------------------
