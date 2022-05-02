///@file nabla_operator.h
///@brief Operator that computes the forward differences along all dimensions
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 09.07.2018

#include "ioperator.h"

namespace optox
{
/**
 * @class NablaOperator
 * @brief Computes first-order differences.
 * 
 * - ``forward``: forward differences
 * - ``adjoint``: backward differences
 */
template <typename T, unsigned int N>
class OPTOX_DLLAPI NablaOperator : public IOperator
{
  public:
    /** Constructor with optional arguments. 
     * 
     * @param hx spacing in x-dimension
     * @param hy spacing in y-dimension
     * @param hz spacing in z-dimension
     * @param ht spacing in t-dimension
    */
    NablaOperator(const T& hx = 1.0, const T& hy = 1.0, const T& hz = 1.0, const T& ht = 1.0) : IOperator(), hx_(hx), hy_(hy), hz_(hz), ht_(ht)
    {
    }

    /** Destructor */
    virtual ~NablaOperator()
    {
    }

    /** No copies are allowed. */
    NablaOperator(NablaOperator const &) = delete;

    /** No assignments are allowed. */
    void operator=(NablaOperator const &) = delete;

  protected:
    /**
     * @brief Compute forward operation by forward differences
     * 
     * - Dimension is given by template parameter `N`
     * - `img_size` denotes the image size
     * 
     * @param outputs Vector of outputs with length 1
     * - [0] Gradient image of size `[N, *img_size]`
     * @param inputs Vector of inputs with length 1
     * - [0] Image of size `img_size`
     */
    virtual void computeForward(OperatorOutputVector &&outputs,
                                const OperatorInputVector &inputs);

    /**
     * @brief Compute adjoint operation (divergence) by backward differences
     * 
     * - Dimension is given by template parameter `N`
     * - `img_size` denotes the image size
     * 
     * @param outputs Vector of outputs with length 1
     * - [0] (Divergence) image of size `img_size`
     * @param inputs Vector of inputs with length 1
     * - [0] Gradient image of size `[N, *img_size]`
     */
    virtual void computeAdjoint(OperatorOutputVector &&outputs,
                                const OperatorInputVector &inputs);

    /** Number of required outputs for the forward operator.
     * @return Number of outputs for forward operator (1)
    */
    virtual unsigned int getNumOutputsForward()
    {
        return 1;
    }

    /** Number of required inputs for the forward operator.
     * @return Number of inputs for forward operator (3)
    */
    virtual unsigned int getNumInputsForward()
    {
        return 1;
    }

    /** Number of required outputs for the adjoint operator.
     * @return Number of outputs for adjoint operator (1)
    */
    virtual unsigned int getNumOutputsAdjoint()
    {
        return 1;
    }

    /** Number of required inputs for the adjoint operator.
     * @return Number of inputs for adjoint operator (1)
    */
    virtual unsigned int getNumInputsAdjoint()
    {
        return 1;
    }

  private:
    T hx_;  /**< spacing in x dimension. */
    T hy_;  /**< spacing in y dimension. */
    T hz_;  /**< spacing in z dimension. */
    T ht_;  /**< spacing in t dimension. */
};

} // namespace optox
