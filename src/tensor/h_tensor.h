///@file h_tensor.h
///@brief Host n-dimensional tensor class for library
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 03.2019

#pragma once

#include "cutils.h"
#include "optox_api.h"
#include "tensor/tensor.h"

#include <string>
#include <type_traits>
#include <initializer_list>

#include <cuda_runtime_api.h>
#include <thrust/host_vector.h>

/** @addtogroup Core 
 *  @{
 */
namespace optox
{

template <typename T, unsigned int N>
class DTensor;

/** 
 *  @class HTensor
 *  @brief Memory management class for host tensors.
 */
template <typename T, unsigned int N>
class OPTOX_DLLAPI HTensor : public Tensor<N>
{
  private:
    T *data_;
    // flag to indicate if a deep copy occurred or not
    bool wrap_;

  public:
    //typedef T type;
    
    /** Constructor. */
    HTensor() : Tensor<N>(), data_(nullptr), wrap_(false)
    {
    }

    /** Special constructor from `optox::Shape`.
    *  @param size Takes tensor size as `optox::Shape` element.
    */
    HTensor(const Shape<N> &size) : Tensor<N>(size), data_(nullptr), wrap_(false)
    {
        data_ = new T[this->numel()];
        if (data_ == nullptr)
            throw std::bad_alloc();
    }

    /** Special constructor from data pointer and `optox::Shape`.
    *  @param data Data pointer
    *  @param size Takes tensor size as `optox::Shape` element.
    *  @param wrap If false, perform deep copy (default). If true, assign `data`.
    */
    HTensor(T *data, const Shape<N> &size, bool wrap = false) : Tensor<N>(size), data_(nullptr), wrap_(wrap)
    {
        if (data == nullptr)
            THROW_OPTOXEXCEPTION("input data not valid");
        if (wrap_)
            data_ = data;
        else
        {
            data_ = new T[this->numel()];
            if (data_ == nullptr)
                throw std::bad_alloc();
            memcpy(data_, data, this->bytes());
        }
    }

    /** No copies are allowed. */
    HTensor(HTensor const &) = delete;

    /** No assignments are allowed. */
    void operator=(HTensor const &) = delete;

    /** Destructor. */
    virtual ~HTensor()
    {
        if ((!wrap_) && (data_ != nullptr))
        {
            delete[] data_;
            data_ = nullptr;
        }
    }

    /** Copy data from another host tensor `optox::HTensor`. 
     * @param from `optox::HTensor` where data should be copied from. Requires same size and stride as this tensor.
    */   
    void copyFrom(const HTensor<T, N> &from)
    {
        if (this->size() != from.size() || this->stride() != from.stride())
            THROW_OPTOXEXCEPTION("Invalid size or stride in copy!");

        memcpy(this->data_, from.constptr(), this->bytes());
    }

    /** Copy data from another device tensor `optox::DTensor`. 
     * @param from `optox::DTensor` where data should be copied from. Requires same size and stride as this tensor.
    */
    void copyFrom(const DTensor<T, N> &from)
    {
        if (this->size() != from.size() || this->stride() != from.stride())
            THROW_OPTOXEXCEPTION("Invalid size or stride in copy!");

        OPTOX_CUDA_SAFE_CALL(
            cudaMemcpy(this->data_, from.constptr(), this->bytes(), cudaMemcpyDeviceToHost));
    }

    /** Fill tensor with a constant scalar (Thrust). 
     * @param scalar fill value.
    */
    template <typename T2>
    void fill(T2 scalar)
    {
        thrust::fill(this->begin(), this->end(), static_cast<T>(scalar));
    }

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
     * @return Thrust pointer.
    */
    thrust::pointer<T, thrust::host_system_tag> begin(void)
    {
        return thrust::pointer<T, thrust::host_system_tag>(ptr());
    }

    /** Point to end of data (Thrust). 
     * @return Thrust pointer.
    */
    thrust::pointer<T, thrust::host_system_tag> end(void)
    {
        return thrust::pointer<T, thrust::host_system_tag>(ptr() + this->numel());
    }

    /** Point to begin of data (Thrust, const). 
     * @return Const Thrust pointer.
    */
    const thrust::pointer<const T, thrust::host_system_tag> begin(void) const
    {
        return thrust::pointer<const T, thrust::host_system_tag>(constptr());
    }

    /** Point to end of data (Thrust, const). 
     * @return Const Thrust pointer.
    */
    const thrust::pointer<const T, thrust::host_system_tag> end(void) const
    {
        return thrust::pointer<const T, thrust::host_system_tag>(constptr() + this->numel());
    }

    /** Returns the state of the `optox::HTensor`.
     * @return False (host)
     */
    virtual bool onDevice() const
    {
        return false;
    }

    // Struct are only used for device code.
    // struct Ref
    // {
    //   private:
    //     T *data_;
    //     const Shape<N> stride_;

    //   public:
    //     const Shape<N> size_;

    //     Ref(HTensor<T, N> &t) : data_(t.ptr()), stride_(t.stride()), size_(t.size())
    //     {
    //     }

    //     ~Ref()
    //     {
    //     }

    //     T &operator()(std::initializer_list<size_t> list)
    //     {
    //         return data_[computeIndex(list)];
    //     }

    //     template <typename T2, class = typename std::enable_if<std::is_integral<T2>{} && (N == 1)>::type>
    //     T &operator()(T2 i)
    //     {
    //         static_assert(N == 1, "wrong access for 1dim Tensor");
    //         return data_[i * stride_[0]];
    //     }

    //     template <typename T2, class = typename std::enable_if<std::is_integral<T2>{} && (N == 2)>::type>
    //     T &operator()(T2 i, T2 j)
    //     {
    //         static_assert(N == 2, "wrong access for 2dim Tensor");
    //         return data_[i * stride_[0] + j * stride_[1]];
    //     }

    //     template <typename T2, class = typename std::enable_if<std::is_integral<T2>{} && (N == 3)>::type>
    //     T &operator()(T2 i, T2 j, T2 k)
    //     {
    //         static_assert(N == 3, "wrong access for 3dim Tensor");
    //         return data_[i * stride_[0] + j * stride_[1] + k * stride_[2]];
    //     }

    //     template <typename T2, class = typename std::enable_if<std::is_integral<T2>{} && (N == 4)>::type>
    //     T &operator()(T2 i, T2 j, T2 k, T2 l)
    //     {
    //         static_assert(N == 4, "wrong access for 4dim Tensor");
    //         return data_[i * stride_[0] + j * stride_[1] + k * stride_[2] + l * stride_[3]];
    //     }

    //     template <typename T2, class = typename std::enable_if<std::is_integral<T2>{} && (N == 5)>::type>
    //     T &operator()(T2 i, T2 j, T2 k, T2 l, T2 m)
    //     {
    //         static_assert(N == 5, "wrong access for 5dim Tensor");
    //         return data_[i * stride_[0] + j * stride_[1] + k * stride_[2] + l * stride_[3] + m * stride_[4]];
    //     }

    //     template <typename A0, typename... Args, class = typename std::enable_if<std::is_integral<A0>{} && (N > 5)>::type>
    //     T &operator()(A0 a0, Args... args)
    //     {
    //         static_assert(sizeof...(Args) == N - 1, "wrong access for Ndim Tensor");
    //         return (*this)(std::initializer_list<size_t>({size_t(a0), size_t(args)...}));
    //     }

    //   private:
    //     size_t computeIndex(const std::initializer_list<size_t> &list) const
    //     {
    //         size_t idx = 0;
    //         auto e = list.begin();
    //         for (unsigned int i = 0; i < N; ++i)
    //         {
    //             idx += stride_[i] * (*e);
    //             ++e;
    //         }
    //         return idx;
    //     }
    // };

    // struct ConstRef
    // {
    //   private:
    //     const T *data_;
    //     const Shape<N> stride_;

    //   public:
    //     const Shape<N> size_;

    //     ConstRef(const HTensor<T, N> &t) : data_(t.constptr()), stride_(t.stride()), size_(t.size())
    //     {
    //     }

    //     ~ConstRef()
    //     {
    //     }

    //     const T &operator()(std::initializer_list<size_t> list) const
    //     {
    //         return data_[computeIndex(list)];
    //     }

    //     template <typename T2, class = typename std::enable_if<std::is_integral<T2>{} && (N == 1)>::type>
    //     const T &operator()(T2 i) const
    //     {
    //         static_assert(N == 1, "wrong access for 1dim Tensor");
    //         return data_[i * stride_[0]];
    //     }

    //     template <typename T2, class = typename std::enable_if<std::is_integral<T2>{} && (N == 2)>::type>
    //     const T &operator()(T2 i, T2 j) const
    //     {
    //         static_assert(N == 2, "wrong access for 2dim Tensor");
    //         return data_[i * stride_[0] + j * stride_[1]];
    //     }

    //     template <typename T2, class = typename std::enable_if<std::is_integral<T2>{} && (N == 3)>::type>
    //     const T &operator()(T2 i, T2 j, T2 k) const
    //     {
    //         static_assert(N == 3, "wrong access for 3dim Tensor");
    //         return data_[i * stride_[0] + j * stride_[1] + k * stride_[2]];
    //     }

    //     template <typename T2, class = typename std::enable_if<std::is_integral<T2>{} && (N == 4)>::type>
    //     const T &operator()(T2 i, T2 j, T2 k, T2 l) const
    //     {
    //         static_assert(N == 4, "wrong access for 4dim Tensor");
    //         return data_[i * stride_[0] + j * stride_[1] + k * stride_[2] + l * stride_[3]];
    //     }

    //     template <typename T2, class = typename std::enable_if<std::is_integral<T2>{} && (N == 5)>::type>
    //     const T &operator()(T2 i, T2 j, T2 k, T2 l, T2 m) const
    //     {
    //         static_assert(N == 4, "wrong access for 5dim Tensor");
    //         return data_[i * stride_[0] + j * stride_[1] + k * stride_[2] + l * stride_[3] + m * stride_[4]];
    //     }

    //     template <typename A0, typename... Args, class = typename std::enable_if<std::is_integral<A0>{} && (N > 5)>::type>
    //     const T &operator()(A0 a0, Args... args) const
    //     {
    //         static_assert(sizeof...(Args) == N - 1, "size missmatch");
    //         return (*this)(std::initializer_list<size_t>({size_t(a0), size_t(args)...}));
    //     }

    //   private:
    //     size_t computeIndex(const std::initializer_list<size_t> &list) const
    //     {
    //         size_t idx = 0;
    //         auto e = list.begin();
    //         for (unsigned int i = 0; i < N; ++i)
    //         {
    //             idx += stride_[i] * (*e);
    //             ++e;
    //         }
    //         return idx;
    //     }
    // };
};

} // namespace optox
/** @}*/
