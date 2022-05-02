///@file scale_operator.h
///@brief Scale Operator that scales an input by a constant value
///@author Kerstin Hammernik k.hammernik@tum.de
///@date 03.2022

#include "ioperator.h"
#include "typetraits.h"

namespace optox
{
/**
 * @class ScaleOperator
 * @brief Scales the input by a constant value
 */
template <typename T>
class OPTOX_DLLAPI ScaleOperator : public IOperator
{
  public:
    /** Constructor. */
    ScaleOperator() : IOperator()
    {
    }

    /** Destructor. */
    virtual ~ScaleOperator()
    {
    }

    /** No copies are allowed. */
    ScaleOperator(ScaleOperator const &) = delete;

    /** No assignments are allowed. */
    void operator=(ScaleOperator const &) = delete;

  protected:
    /**
     * @brief Compute forward scaling for 2D data.
     * 
     * - `N`: batch size
     * - `W`: width of the image
     * 
     * @param outputs Vector of outputs with length 1
     * - [0] Scaled input `[N, W]`
     * @param inputs Vector of inputs with length 1
     * - [0] Image of size `[N, W]`
     */
    virtual void computeForward(OperatorOutputVector &&outputs,
                                const OperatorInputVector &inputs);

    /**
     * @brief Compute adjoint scaling for 2D data. This equals forward scaling.
     * 
     * - `N`: batch size
     * - `W`: width of the image
     * 
     * @param outputs Vector of outputs with length 1
     * - [0] Scaled input `[N, W]`
     * @param inputs Vector of inputs with length 1
     * - [0] Image of size `[N, W]`
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
     * @return Number of inputs for forward operator (1)
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
};

} // namespace optox
