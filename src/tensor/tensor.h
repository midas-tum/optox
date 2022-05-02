///@file tensor.h
///@brief Basic n-dimensional tensor class for library
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 03.2019

#pragma once

#include "optox_api.h"
#include "tensor/shape.h"

/** @defgroup Core Core group
 *  @brief Defines core functionality for library.
 * 
 * - Shape
 * - Tensor
 *     - Tensor on host `HTensor`
 *     - Tensor on device `DTensor`
 * - IOperator
 * @{
 */
namespace optox
{
/**
 * @class ITensor
 * @brief Interface for operators
 *  `ITensor` defines the interface for all tensors.
 */
class OPTOX_DLLAPI ITensor
{
  public:
    /** Constructor. */
    ITensor()
    {
    }

    /** Destructor. */
    virtual ~ITensor()
    {
    }

    /** No copies are allowed. */
    ITensor(ITensor const &) = delete;

    /** No assignments are allowed. */
    void operator=(ITensor const &) = delete;

    /** Returns the state of the tensor.
     * @return True if the tensor is allocated on Device, False if it is allocated on Host.
     */
    virtual bool onDevice() const = 0;

    /** Returns the tensor's allocation in bytes.
     * @return The tensor's allocation in bytes.
     */
    virtual size_t bytes() const = 0;
};

/** @class Tensor
 *  @brief Defines common tensor operations for all tensor types.
 * 
 *  - tensor on host `optox::HTensor`
 *  - tensor on device `optox::DTensor`
 * 
 */
template <unsigned int N>
class OPTOX_DLLAPI Tensor : public ITensor
{
  protected:
    Shape<N> size_;  /** Size of the tensor data. */
    Shape<N> stride_;  /** Stride of the memory. */

    void computeStride()
    {
        stride_[N-1] = 1;
        for (int i = N - 2; i >= 0; --i)
            stride_[i] = stride_[i + 1] * size_[i + 1];
    }

  public:
    /** Constructor. */
    Tensor() : ITensor(), size_(), stride_()
    {
    }

    /** Special constructor from `optox::Shape`.
    *  @param size Takes tensor size as `optox::Shape` element.
    */
    Tensor(const Shape<N> &size) : ITensor(), size_(size)
    {
        computeStride();
    }

    /** Destructor. */
    virtual ~Tensor()
    {
    }

    /** No copies are allowed. */
    Tensor(Tensor const &) = delete;

    /** No assignments are allowed. */
    void operator=(Tensor const &) = delete;

    /** Returns the number of elements saved in the buffer. */
    size_t numel() const
    {
        return size_.numel();
    }

    /** Returns the tensor's size as `optox::Shape`. */
    const Shape<N> &size() const
    {
        return size_;
    }

    /** Returns the tensor's stride as `optox::Shape`. */
    const Shape<N> &stride() const
    {
        return stride_;
    }

    /** Operator<< overloading to define output of tensor class. */
    friend std::ostream &operator<<(std::ostream &out, Tensor const &t)
    {
        out << "Tensor: size=" << t.size() << " strides="
            << t.stride() << " numel=" << t.numel() << " onDevice="
            << t.onDevice();
        return out;
    }
};
} // namespace optox
/**@}*/ // End of Doxygen group Core
