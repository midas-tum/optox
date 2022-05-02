///@file utils.h
///@brief utility functions
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 03.2019

#pragma once

#include <iostream>
#include <sstream>

#if defined(_MSC_VER)
#define __FORCEINLINE__ __forceinline
#else
#define __FORCEINLINE__ __attribute__((always_inline)) inline
#endif

#ifdef __CUDACC__
#include <cuda_runtime_api.h>
#undef __HOSTDEVICE__
#undef __HOST__
#undef __DEVICE__
#undef __DHOSTDEVICE__
#define __HOSTDEVICE__ __host__ __device__ __forceinline__
#define __HOST__ __host__ __forceinline__
#define __DEVICE__ __device__ __forceinline__
#define __RESTRICT__ __restrict__
#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION < 9000)
#define __DHOSTDEVICE__ __host__ __device__ __forceinline__
#elif (CUDART_VERSION >= 9000)
#define __DHOSTDEVICE__ __forceinline__
#endif
#else
#undef __HOSTDEVICE__
#undef __HOST__
#undef __DEVICE__
#define __HOSTDEVICE__ __FORCEINLINE__
#define __DHOSTDEVICE__ __FORCEINLINE__
#define __HOST__ __FORCEINLINE__
#define __DEVICE__ __FORCEINLINE__
#define __RESTRICT__
#endif


#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

class OpToXException : public std::exception
{
  public:
    /** Special constructor.
     *  
     * @param msg Exception message
     * @param file File where exception waw thrown (default: null)
     * @param function Function where exception was thrown (default: null)
     * @param line Line where exception was thrown (default: null)
     */
    OpToXException(const std::string &msg, const char *file = nullptr, const char *function = nullptr, int line = 0) throw() : msg_(msg),
                                                                                                                               file_(file),
                                                                                                                               function_(function),
                                                                                                                               line_(line)
    {
        std::ostringstream out_msg;

        out_msg << "OpToXException: ";
        out_msg << (msg_.empty() ? "unknown error" : msg_) << "\n";
        out_msg << "      where: ";
        out_msg << (file_.empty() ? "no filename available" : file_) << " | ";
        out_msg << (function_.empty() ? "unknown function" : function_) << ":" << line_;
        msg_ = out_msg.str();
    }

    /** Destructor. */
    virtual ~OpToXException() throw()
    {
    }

    /**
     * @brief Return explanation
     * 
     * @return const char* explanation
     */
    virtual const char *what() const throw()
    {
        return msg_.c_str();
    }

  private:
    std::string msg_;
    std::string file_;
    std::string function_;
    int line_;
};

#define THROW_OPTOXEXCEPTION(str) throw OpToXException(str, __FILE__, __FUNCTION__, __LINE__)

namespace optox
{

static inline unsigned int divUp(unsigned int a, unsigned int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

}
