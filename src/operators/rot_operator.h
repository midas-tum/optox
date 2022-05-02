///@file rot_operator.h
///@brief Operator rotating kernel stack
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 06.2019

#include "ioperator.h"

namespace optox
{

template <typename T>
class OPTOX_DLLAPI RotOperator : public IOperator
{
  public:
    /** Constructor. */
    RotOperator() : IOperator()
    {
    }

    /** Destructor. */
    virtual ~RotOperator()
    {
    }

    /** No copies are allowed. */
    RotOperator(RotOperator const &) = delete;

    /** No assignments are allowed. */
    void operator=(RotOperator const &) = delete;

  protected:
    /**
     * @brief Compute forward rotation of input by a given set of rotation angles.
     * 
     * - `N`: batch size
     * - `Nangles`: number of angles
     * - `H`: height of the image
     * - `W`: width of the image
     * - `C`: feature channels
     * 
     * @param outputs Vector of outputs with length 1
     * - [0] output with rotated versions of input, size `[N, Nangles, C, H, W]`
     * 
     * @param inputs Vector of inputs with length 2
     * - [0] input of size `[N, C, H, W]`
     * - [1] array of angles `[Nangles]`
     */
    virtual void computeForward(OperatorOutputVector &&outputs,
                                const OperatorInputVector &inputs);

    /**
     * @brief Compute adjoint rotation of input wrt. given set of rotation angles.
     * 
     * - `N`: batch size
     * - `Nangles`: number of angles
     * - `H`: height of the image
     * - `W`: width of the image
     * - `C`: feature channels
     * 
     * @param outputs Vector of outputs with length 1
     * - [0] (back-polated) output of size `[N, C, H, W]`
     * 
     * @param inputs Vector of inputs with length 2
     * - [0] input of size `[N, Nangles, C, H, W]`
     * - [1] array of angles `[Nangles]`
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
     * @return Number of inputs for forward operator (2)
    */
    virtual unsigned int getNumInputsForward()
    {
        return 2;
    }

    /** Number of required outputs for the adjoint operator.
     * @return Number of outputs for adjoint operator (1)
    */
    virtual unsigned int getNumOutputsAdjoint()
    {
        return 1;
    }

    /** Number of required inputs for the adjoint operator.
     * @return Number of inputs for adjoint operator (2)
    */
    virtual unsigned int getNumInputsAdjoint()
    {
        return 2;
    }
};

} // namespace optox
