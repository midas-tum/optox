///@file ioperator.h
///@brief Interface for operators
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 09.07.2018

#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <sstream>

#include <initializer_list>
#include <vector>
#include <map>

#include "optox_api.h"
#include "utils.h"
#include "tensor/tensor.h"
#include "tensor/d_tensor.h"

#include <cuda.h>

/** \addtogroup Core 
 *  @{
 */
namespace optox
{

typedef std::vector<const ITensor *> OperatorInputVector;
typedef std::vector<ITensor *> OperatorOutputVector;

/**
 * \class IOperator
 * \brief Interface for operators
 * 
 *  `IOperator` defines 
 *      - the common functions that *all* operators must implement.
 *      - auxiliary helper functions
 */

class OPTOX_DLLAPI IOperator
{
  public:
    /** Constructor. */
    IOperator() : stream_(cudaStreamDefault)
    {
    }

    /** Destructor. */
    virtual ~IOperator()
    {
    }

    /** Apply the operator's forward.
     * @param outputs list of the operator's forward outputs `{&out_1, ...}` which are
     *          typically of type `DTensor`
     * @param inputs list of the operator's forward inputs `{&in_1, ...}` which are
     *         typically of type `DTensor`
     */
    void forward(std::initializer_list<ITensor *> outputs,
                 std::initializer_list<const ITensor *> inputs)
    {
        if (outputs.size() != getNumOutputsForward())
            THROW_OPTOXEXCEPTION("Provided number of outputs does not match the required number!");

        if (inputs.size() != getNumInputsForward())
            THROW_OPTOXEXCEPTION("Provided number of inputs does not match the required number!");

        computeForward(OperatorOutputVector(outputs), OperatorInputVector(inputs));
    }

    /** Apply the operator's adjoint.
     * @param outputs list of the operator's adjoint outputs `{&out_1, ...}` which are
     *          typically of type `DTensor`
     * @param inputs list of the operator's adjoint inputs `{&in_1, ...}` which are
     *         typically of type `DTensor`
     */
    void adjoint(std::initializer_list<ITensor *> outputs,
                 std::initializer_list<const ITensor *> inputs)
    {
        if (outputs.size() != getNumOutputsAdjoint())
            THROW_OPTOXEXCEPTION("Provided number of outputs does not match the requied number!");

        if (inputs.size() != getNumInputsAdjoint())
            THROW_OPTOXEXCEPTION("Provided number of inputs does not match the requied number!");

        computeAdjoint(OperatorOutputVector(outputs), OperatorInputVector(inputs));
    }

    /** Get the input `DTensor` at given `index`.
     * @param index Index of input tensor in `inputs` to return
     * @param inputs Vector of input tensors
     */
    template <typename T, unsigned int N>
    const DTensor<T, N> *getInput(unsigned int index, const OperatorInputVector &inputs)
    {
        if (index < inputs.size())
        {
            const DTensor<T, N> *t = dynamic_cast<const DTensor<T, N> *>(inputs[index]);
            if (t != nullptr)
                return t;
            else
                THROW_OPTOXEXCEPTION("Cannot cast input to desired type!");
        }
        else
            THROW_OPTOXEXCEPTION("input index out of bounds!");
    }

    /** Get the output `DTensor` at given `index`.
     * @param index Index of output tensor in `outputs` to return
     * @param outputs Vector of output tensors
     */
    template <typename T, unsigned int N>
    DTensor<T, N> *getOutput(unsigned int index, const OperatorOutputVector &outputs)
    {
        if (index < outputs.size())
        {
            DTensor<T, N> *t = dynamic_cast<DTensor<T, N> *>(outputs[index]);
            if (t != nullptr)
                return t;
            else
                THROW_OPTOXEXCEPTION("Cannot cast output to desired type!");
        }
        else
            THROW_OPTOXEXCEPTION("output index out of bounds!");
    }

    /** Set CUDA stream that should be used to call the CUDA kernels.
     * @param stream CUDA stream
    */
    void setStream(const cudaStream_t &stream)
    {
        stream_ = stream;
    }

    /** Get CUDA stream that is used to call the CUDA kernels.
     * @return CUDA stream
     */
    cudaStream_t getStream() const
    {
        return stream_;
    }

    /** No copies are allowed. */
    IOperator(IOperator const &) = delete;

    /** No assignments are allowed. */
    void operator=(IOperator const &) = delete;

  protected:
    /** Actual implementation of the forward operator.
     * @param outputs Vector of outputs that are computed by the forward operator
     * @param inputs Vector of inputs that are required to compute
     */
    virtual void computeForward(OperatorOutputVector &&outputs,
                                const OperatorInputVector &inputs) = 0;

    /** Actual implementation of the adjoint operator.
     * @param outputs Vector of outputs that are computed by the forward operator
     * @param inputs Vector of inputs that are required to compute
     */
    virtual void computeAdjoint(OperatorOutputVector &&outputs,
                                const OperatorInputVector &inputs) = 0;

    /** Number of required outputs for the forward operator.
     * @return Number of outputs for forward operator
    */
    virtual unsigned int getNumOutputsForward() = 0;
    /** Number of required inputs for the forward operator.
     * @return Number of inputs for forward operator
    */
    virtual unsigned int getNumInputsForward() = 0;

    /** Number of required outputs for the adjoint operator.
     * @return Number of outputs for adjoint operator
    */
    virtual unsigned int getNumOutputsAdjoint() = 0;
    /** Number of required inputs for the adjoint operator.
     * @return Number of inputs for adjoint operator
    */
    virtual unsigned int getNumInputsAdjoint() = 0;

  protected:
    /** CUDA stream to be used to call the CUDA kernels. */
    cudaStream_t stream_;
};

#define OPTOX_CALL_float(m) m(float)
#define OPTOX_CALL_double(m) m(double)
#define OPTOX_CALL_float2(m) m(float2)
#define OPTOX_CALL_double2(m) m(double2)

#define OPTOX_CALL_REAL_NUMBER_TYPES(m) \
    OPTOX_CALL_float(m) OPTOX_CALL_double(m)
#define OPTOX_CALL_NUMBER_TYPES(m) \
    OPTOX_CALL_float(m) OPTOX_CALL_double(m) OPTOX_CALL_float2(m) OPTOX_CALL_double2(m)
} // namespace optox
/** @}*/ // End of Doxygen group Core
