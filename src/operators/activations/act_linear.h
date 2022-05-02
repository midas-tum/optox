///@file act_linear.h
///@brief linear interpolation activation function operator
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 01.2019

#include "act.h"

namespace optox
{

/**
 * @class LinearActOperator
 * @brief Activation operator based on linear interpolation
 */
template <typename T>
class OPTOX_DLLAPI LinearActOperator : public IActOperator<T>
{
  public:
    /** Constructor */
    LinearActOperator(T vmin, T vmax) : IActOperator<T>(vmin, vmax)
    {
    }

    /** Destructor */
    virtual ~LinearActOperator()
    {
    }

    /** No copies are allowed. */
    LinearActOperator(LinearActOperator const &) = delete;
    
    /** No assignments are allowed. */
    void operator=(LinearActOperator const &) = delete;

  protected:
    virtual void computeForward(OperatorOutputVector &&outputs,
                                const OperatorInputVector &inputs);

    virtual void computeAdjoint(OperatorOutputVector &&outputs,
                                const OperatorInputVector &inputs);
};

} // namespace optox
