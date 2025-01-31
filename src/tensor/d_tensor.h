///@file d_tensor.h
///@brief Device n-dimensional tensor class for library
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 03.2019

#pragma once

#include "utils.h"
#include "cutils.h"
#include "optox_api.h"
#include "tensor/tensor.h"

#include <string>
#include <type_traits>
#include <initializer_list>

#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>

/** @addtogroup Core
 *  @{
 */
namespace optox
{

template <typename T, unsigned int N>
class HTensor;

/** 
 *  @class DTensor
 *  @brief Memory management class for device tensors.
 */
template <typename T, unsigned int N>
class OPTOX_DLLAPI DTensor : public Tensor<N>
{
  private:
    T *data_;
    // flag to indicate if a deep copy occurs or not
    bool wrap_;

  public:
    //typedef T type;

    /** Constructor. */
    DTensor() : Tensor<N>(), data_(nullptr), wrap_(false)
    {
    }

    /** Special constructor from `optox::Shape`.
    *  @param size Takes tensor size as `optox::Shape` element.
    */
    DTensor(const Shape<N> &size) : Tensor<N>(size), data_(nullptr), wrap_(false)
    {
        OPTOX_CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&data_), this->bytes()));
        if (data_ == nullptr)
            throw std::bad_alloc();
    }

    /** Special constructor from data pointer and `optox::Shape`.
    *  @param data Data pointer
    *  @param size Takes tensor size as `optox::Shape` element.
    *  @param wrap If false, perform deep copy (default). If true, assign `data`.
    */
    DTensor(T *data, const Shape<N> &size, bool wrap = false) : Tensor<N>(size), data_(nullptr), wrap_(wrap)
    {
        if (data == nullptr)
            THROW_OPTOXEXCEPTION("input data not valid");
        if (wrap_)
            data_ = data;
        else
        {
            OPTOX_CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&data_), this->bytes()));
            if (data_ == nullptr)
                throw std::bad_alloc();
            OPTOX_CUDA_SAFE_CALL(cudaMemcpy(this->data_, data, this->bytes(), cudaMemcpyDeviceToDevice));
        }
    }

    /** No copies are allowed. */
    DTensor(DTensor const &) = delete;

    /** No assignments are allowed. */
    void operator=(DTensor const &) = delete;

    /** Destructor. */
    virtual ~DTensor()
    {
        if ((!wrap_) && (data_ != nullptr))
        {
            OPTOX_CUDA_SAFE_CALL(cudaFree(data_));
            data_ = nullptr;
        }
    }

    /** Copy data from another host pointer. 
     * @param from host pointer.
    */   
    void copyFromHostPtr(const T *from)
    {
        OPTOX_CUDA_SAFE_CALL(
            cudaMemcpy(this->data_, from, this->bytes(), cudaMemcpyHostToDevice));
    }

    /** Copy data from another device tensor `optox::DTensor`. 
     * @param from `optox::DTensor` where data should be copied from. Requires same size and stride as this tensor.
    */   
    void copyFrom(const DTensor<T, N> &from)
    {
        if (this->size() != from.size() || this->stride() != from.stride())
            THROW_OPTOXEXCEPTION("Invalid size or stride in copy!");

        OPTOX_CUDA_SAFE_CALL(
            cudaMemcpy(this->data_, from.constptr(), this->bytes(), cudaMemcpyDeviceToDevice));
    }

    /** Copy data from another host tensor `optox::HTensor`. 
     * @param from `optox::HTensor` where data should be copied from. Requires same size and stride as this tensor.
    */   
    void copyFrom(const HTensor<T, N> &from)
    {
        if (this->size() != from.size() || this->stride() != from.stride())
            THROW_OPTOXEXCEPTION("Invalid size or stride in copy!");

        OPTOX_CUDA_SAFE_CALL(
            cudaMemcpy(this->data_, from.constptr(), this->bytes(), cudaMemcpyHostToDevice));
    }

    #ifdef __CUDACC__
    /** Fill tensor with a constant scalar (Thrust). 
     * @param scalar fill value.
    */
    template <typename T2>
    void fill(T2 scalar, const cudaStream_t stream = nullptr)
    {
        if (stream != nullptr)
            thrust::fill(thrust::cuda::par.on(stream), this->begin(), this->end(), static_cast<T>(scalar));
        else
            thrust::fill(this->begin(), this->end(), static_cast<T>(scalar));
    }
    #endif

    /** Returns the tensor's allocation in bytes.
     * @return The tensor's allocation in bytes.
     */
    size_t bytes() const
    {
        return this->numel() * sizeof(T);
    }

    /** Get data pointer. 
     * @param offset Get data pointer starting at offset.
    */
    T *ptr(unsigned int offset = 0)
    {
        if (offset >= this->numel())
        {
            std::stringstream msg;
            msg << "Index (" << offset << ") out of range (" << this->numel() << ").";
            THROW_OPTOXEXCEPTION(msg.str());
        }
        return &(data_[offset]);
    }

    /** Get const data pointer. 
     * @param offset Get const data pointer starting at offset.
    */
    const T *constptr(unsigned int offset = 0) const
    {
        if (offset >= this->numel())
        {
            std::stringstream msg;
            msg << "Offset (" << offset << ") out of range (" << this->numel() << ").";
            THROW_OPTOXEXCEPTION(msg.str());
        }
        return reinterpret_cast<const T *>(&(data_[offset]));
    }

    /** Point to begin of data (Thrust). 
     * @return Thrust device pointer.
    */
    thrust::device_ptr<T> begin(void)
    {
        return thrust::device_ptr<T>(ptr());
    }

    /** Point to end of data (Thrust). 
     * @return Thrust device pointer.
    */
    thrust::device_ptr<T> end(void)
    {
        return thrust::device_ptr<T>(ptr() + this->numel());
    }

    /** Point to begin of data (Thrust, const). 
     * @return Const Thrust device pointer.
    */
    const thrust::device_ptr<T> begin(void) const
    {
        return thrust::device_ptr<T>(constptr());
    }

    /** Point to end of data (Thrust, const). 
     * @return Const Thrust device pointer.
    */
    const thrust::device_ptr<T> end(void) const
    {
        return thrust::device_ptr<T>(constptr() + this->numel());
    }

    /** Returns the state of the `optox::DTensor`.
     * @return True (device)
     */
    virtual bool onDevice() const
    {
        return true;
    }

    struct Ref
    {
      private:
        T *data_;
        const Shape<N> stride_;

      public:
        const Shape<N> size_;

        __HOST__ Ref(DTensor<T, N> &t) : data_(t.ptr()), stride_(t.stride()), size_(t.size())
        {
        }

        __HOST__ ~Ref()
        {
        }

        __DEVICE__ T &operator()(std::initializer_list<size_t> list)
        {
            return data_[computeIndex(list)];
        }

        // template <typename T2, class = typename std::enable_if<std::is_integral<T2>{} && (N == 1)>::type>
        // __DEVICE__ T &operator()(T2 i)
        // {
        //     static_assert(N == 1, "wrong access for 1dim Tensor");
        //     return data_[i * stride_[0]];
        // }

        // template <typename T2, class = typename std::enable_if<std::is_integral<T2>{} && (N == 2)>::type>
        // __DEVICE__ T &operator()(T2 i, T2 j)
        // {
        //     static_assert(N == 2, "wrong access for 2dim Tensor");
        //     return data_[i * stride_[0] + j * stride_[1]];
        // }

        // template <typename T2, class = typename std::enable_if<std::is_integral<T2>{} && (N == 3)>::type>
        // __DEVICE__ T &operator()(T2 i, T2 j, T2 k)
        // {
        //     static_assert(N == 3, "wrong access for 3dim Tensor");
        //     return data_[i * stride_[0] + j * stride_[1] + k * stride_[2]];
        // }

        // template <typename T2, class = typename std::enable_if<std::is_integral<T2>{} && (N == 4)>::type>
        // __DEVICE__ T &operator()(T2 i, T2 j, T2 k, T2 l)
        // {
        //     static_assert(N == 4, "wrong access for 4dim Tensor");
        //     return data_[i * stride_[0] + j * stride_[1] + k * stride_[2] + l * stride_[3]];
        // }

        // template <typename T2, class = typename std::enable_if<std::is_integral<T2>{} && (N == 5)>::type>
        // __DEVICE__ T &operator()(T2 i, T2 j, T2 k, T2 l, T2 m)
        // {
        //     static_assert(N == 5, "wrong access for 5dim Tensor");
        //     return data_[i * stride_[0] + j * stride_[1] + k * stride_[2] + l * stride_[3] + m * stride_[4]];
        // }

        template <typename A0, typename... Args, class = typename std::enable_if<std::is_integral<A0>{}>::type>
        __DEVICE__ T &operator()(A0 a0, Args... args)
        {
            static_assert(sizeof...(Args) == N - 1, "wrong access for Ndim Tensor");
            return (*this)(std::initializer_list<size_t>({size_t(a0), size_t(args)...}));
        }

      private:
        __DEVICE__ size_t computeIndex(const std::initializer_list<size_t> &list) const
        {
            size_t idx = 0;
            auto e = list.begin();
            for (unsigned int i = 0; i < N; ++i)
            {
                idx += stride_[i] * (*e);
                ++e;
            }
            return idx;
        }
    };

    struct ConstRef
    {
      private:
        const T *data_;
        const Shape<N> stride_;

      public:
        const Shape<N> size_;

        __HOST__ ConstRef(const DTensor<T, N> &t) : data_(t.constptr()), stride_(t.stride()), size_(t.size())
        {
        }

        __HOST__ ~ConstRef()
        {
        }

        __DEVICE__ const T &operator()(std::initializer_list<size_t> list) const
        {
            return data_[computeIndex(list)];
        }

        // template <typename T2, class = typename std::enable_if<std::is_integral<T2>{} && (N == 1)>::type>
        // __DEVICE__ const T &operator()(T2 i) const
        // {
        //     static_assert(N == 1, "wrong access for 1dim Tensor");
        //     return data_[i * stride_[0]];
        // }

        // template <typename T2, class = typename std::enable_if<std::is_integral<T2>{} && (N == 2)>::type>
        // __DEVICE__ const T &operator()(T2 i, T2 j) const
        // {
        //     static_assert(N == 2, "wrong access for 2dim Tensor");
        //     return data_[i * stride_[0] + j * stride_[1]];
        // }

        // template <typename T2, class = typename std::enable_if<std::is_integral<T2>{} && (N == 3)>::type>
        // __DEVICE__ const T &operator()(T2 i, T2 j, T2 k) const
        // {
        //     static_assert(N == 3, "wrong access for 3dim Tensor");
        //     return data_[i * stride_[0] + j * stride_[1] + k * stride_[2]];
        // }

        // template <typename T2, class = typename std::enable_if<std::is_integral<T2>{} && (N == 4)>::type>
        // __DEVICE__ const T &operator()(T2 i, T2 j, T2 k, T2 l) const
        // {
        //     static_assert(N == 4, "wrong access for 4dim Tensor");
        //     return data_[i * stride_[0] + j * stride_[1] + k * stride_[2] + l * stride_[3]];
        // }

        // template <typename T2, class = typename std::enable_if<std::is_integral<T2>{} && (N == 5)>::type>
        // __DEVICE__ const T &operator()(T2 i, T2 j, T2 k, T2 l, T2 m) const
        // {
        //     static_assert(N == 5, "wrong access for 5dim Tensor");
        //     return data_[i * stride_[0] + j * stride_[1] + k * stride_[2] + l * stride_[3] + m * stride_[4]];
        // }

        template <typename A0, typename... Args, class = typename std::enable_if<std::is_integral<A0>{}>::type>
        __DEVICE__ const T &operator()(A0 a0, Args... args) const
        {
            static_assert(sizeof...(Args) == N - 1, "size missmatch");
            return (*this)(std::initializer_list<size_t>({size_t(a0), size_t(args)...}));
        }

      private:
        __DEVICE__ size_t computeIndex(const std::initializer_list<size_t> &list) const
        {
            size_t idx = 0;
            auto e = list.begin();
            for (unsigned int i = 0; i < N; ++i)
            {
                idx += stride_[i] * (*e);
                ++e;
            }
            return idx;
        }
    };
};

} // namespace optox
/** @}*/ // End of Doxygen group Core
