///@file act.h
///@brief Operator for basic activation function interface
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 01.2019

#pragma once

#include "tensor/shape.h"
#include "operators/ioperator.h"

namespace optox
{
/**
 * @class IActOperator
 * @brief Interface class for activation operators
 * 
 * This class defines the interface for trainable activation operators.
 * Activation functions are defined for a support `[vmin_, vmax_]`.
 */
template <typename T>
class OPTOX_DLLAPI IActOperator : public IOperator
{
  public:
    /** Constructor. */
    IActOperator(T vmin, T vmax) : IOperator(), vmin_(vmin), vmax_(vmax)
    {
    }

    /** Destructor. */
    virtual ~IActOperator()
    {
    }

    /** No copies are allowed. */
    IActOperator(IActOperator const &) = delete;

    /** No assignments are allowed. */
    void operator=(IActOperator const &) = delete;

  protected:
    /**
     * @brief Compute forward pass. Needs to be implemented in derived classes.
     * 
     * - `C`: number of receive coils
     * - `N`: number of frequency encoding points
     * - `Nw`: number of weights to form activation function
     * 
     * @param outputs Vector of outputs with length 1
     * - [0] Result after pixel-wise application of activation `[C, N]`
     * @param inputs Vector of inputs with length 2
     * - [0] Input of size `[C, N]`
     * - [1] Weights of size `[C, Nw]`
     */
    virtual void computeForward(OperatorOutputVector &&outputs,
                                const OperatorInputVector &inputs) = 0;

    /**
     * @brief Compute adjoint pass. Needs to be implemented in derived classes.
     * 
     * @param outputs Vector of outputs with length 1
     * - [0] Gradient wrt. input `inputs[0].shape=[C, N]`
     * - [1] Gradient wrt. to weights `inputs[1].shape=[C, Nw]`
     * @param inputs Vector of inputs with length 2
     * - [0] Input of size `[C, N]`
     * - [1] Weights of size `[C, Nw]`
     * - [2] Gradient wrt. output of size `[C, N]`
     */
    virtual void computeAdjoint(OperatorOutputVector &&outputs,
                                const OperatorInputVector &inputs) = 0;


    /**
     * @brief Check if sizes of input and weights match along the channel (C) dimension.
     * 
     * @param input_size optox::Shape of input
     * @param weights_size optox::Shape of weights
     */
    void checkSize(const Shape<2> input_size, const Shape<2> weights_size)
    {
        if (input_size[0] != weights_size[0])
            throw std::runtime_error("Activation operator: input and weights size do not match!");
    }

    /** Number of required outputs for the forward operator.
     * @return Number of outputs for forward operator (1)
    */
    virtual unsigned int getNumOutputsForward()
    {
        return 1;
    }

    /** Number of required inputs for the forward operator.
     * @return Number of inputs for forward operator (2)
    */
    virtual unsigned int getNumInputsForward()
    {
        return 2;
    }

    /** Number of required outputs for the adjoint operator.
     * @return Number of outputs for adjoint operator (2)
    */
    virtual unsigned int getNumOutputsAdjoint()
    {
        return 2;
    }

    /** Number of required inputs for the adjoint operator.
     * @return Number of inputs for adjoint operator (3)
    */
    virtual unsigned int getNumInputsAdjoint()
    {
        return 3;
    }

  protected:
    /** Min support of activation functions. */
    T vmin_;
    /** Max support of activation functions. */
    T vmax_;
};

/**
 * @class IAct2Operator
 * @brief Interface class for activation operators, returning also the first derivative of the activation.
 * 
 * This class defines the interface for trainable activation operators.
 * Activation functions are defined for a support `[vmin_, vmax_]`.
 * Outputs are the applied activation, and the first derivative of the applied activation.
 */

template <typename T>
class OPTOX_DLLAPI IAct2Operator : public IOperator
{
  public:
    /** Constructor. */
    IAct2Operator(T vmin, T vmax) : IOperator(), vmin_(vmin), vmax_(vmax)
    {
    }

    /** Destructor. */
    virtual ~IAct2Operator()
    {
    }

    /** No copies are allowed. */
    IAct2Operator(IAct2Operator const &) = delete;

    /** No assignments are allowed. */
    void operator=(IAct2Operator const &) = delete;

  protected:
    /**
     * @brief Compute forward pass. Needs to be implemented in derived classes.
     * 
     * - `C`: number of receive coils
     * - `N`: number of frequency encoding points
     * - `Nw`: number of weights to form activation function
     * 
     * @param outputs Vector of outputs with length 1
     * - [0] Result after pixel-wise application of activation `[C, N]`
     * - [1] Result after pixel-wise application of first-order derivative of activation `[C, N]`
     * @param inputs Vector of inputs with length 2
     * - [0] Input of size `[C, N]`
     * - [1] Weights of size `[C, Nw]`
     */
    virtual void computeForward(OperatorOutputVector &&outputs,
                                const OperatorInputVector &inputs) = 0;

    /**
     * @brief Compute adjoint pass. Needs to be implemented in derived classes.
     * 
     * @param outputs Vector of outputs with length 1
     * - [0] Gradient wrt. input `inputs[0].shape=[C, N]`
     * - [1] Gradient wrt. to weights `inputs[1].shape=[C, Nw]`
     * @param inputs Vector of inputs with length 2
     * - [0] Input of size `[C, N]`
     * - [1] Weights of size `[C, Nw]`
     * - [2] Gradient wrt. `outputs[0]` of size `[C, N]`
     * - [3] Gradient wrt. `outputs[1]` of size `[C, N]`
     */
    virtual void computeAdjoint(OperatorOutputVector &&outputs,
                                const OperatorInputVector &inputs) = 0;

    /**
     * @brief Check if sizes of input and weights match along the channel (C) dimension.
     * 
     * @param input_size optox::Shape of input
     * @param weights_size optox::Shape of weights
     */
    void checkSize(const Shape<2> input_size, const Shape<2> weights_size)
    {
        if (input_size[0] != weights_size[0])
            throw std::runtime_error("Activation operator: input and weights size do not match!");
    }

    /** Number of required outputs for the forward operator.
     * @return Number of outputs for forward operator (2)
    */
    virtual unsigned int getNumOutputsForward()
    {
        return 2;
    }

    /** Number of required inputs for the forward operator.
     * @return Number of inputs for forward operator (2)
    */
    virtual unsigned int getNumInputsForward()
    {
        return 2;
    }

    /** Number of required outputs for the adjoint operator.
     * @return Number of outputs for adjoint operator (2)
    */
    virtual unsigned int getNumOutputsAdjoint()
    {
        return 2;
    }

    /** Number of required inputs for the adjoint operator.
     * @return Number of inputs for adjoint operator (4)
    */
    virtual unsigned int getNumInputsAdjoint()
    {
        return 4;
    }

  protected:
    /** Min support of activation functions. */
    T vmin_;
    /** Max support of activation functions. */
    T vmax_;
};

} // namespace optox
