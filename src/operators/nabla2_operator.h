///@file nabla2_operator.h
///@brief Operator that computes the second order forward differences along all dimensions
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 02.2019

#include "ioperator.h"

namespace optox
{

template <typename T, unsigned int N>
class OPTOX_DLLAPI Nabla2Operator : public IOperator
{
  public:
    /** Constructor. */
    Nabla2Operator() : IOperator()
    {
    }

    /** Destructor */
    virtual ~Nabla2Operator()
    {
    }

    /** No copies are allowed. */
    Nabla2Operator(Nabla2Operator const &) = delete;

    /** No assignments are allowed. */
    void operator=(Nabla2Operator const &) = delete;

  protected:
    /**
     * @brief Compute forward operation by forward differences
     * 
     * - Dimension is given by template parameter `N`
     * - `img_size` denotes the image size
     * 
     * For 2D, the second-order gradient image contains derivates as follows:
     * - [0] ... xx
     * - [1] ... xy
     * - [2] ... yx
     * - [3] ... yy
     * 
     * @param outputs Vector of outputs with length 1
     * - [0] 2nd-order gradient image of size `[N*N, *img_size]`
     * @param inputs Vector of inputs with length 1
     * - [0] Image of size `img_size`
     */
    virtual void computeForward(OperatorOutputVector &&outputs,
                                const OperatorInputVector &inputs);

    /**
     * @brief Compute forward operation by backward differences
     * 
     * - Dimension is given by template parameter `N`
     * - `img_size` denotes the image size
     * 
     * @param outputs Vector of outputs with length 1
     * - [0] (Divergence) image of size `img_size`
     * @param inputs Vector of inputs with length 1
     * - [0] 2nd-order gradient image of size `[N*N, *img_size]`
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
};

} // namespace optox
