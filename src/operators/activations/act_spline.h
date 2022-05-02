///@file act_spline.h
///@brief Spline basis function operator
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 06.2019

#include "act.h"

namespace optox
{
/**
 * @class SplineActOperator
 * @brief Activation operator based on B-splines 
 */
template <typename T>
class OPTOX_DLLAPI SplineActOperator : public IActOperator<T>
{
  public:
    /** Constructor. */
    SplineActOperator(T vmin, T vmax) : IActOperator<T>(vmin, vmax)
    {
    }

    /** Destructor. */
    virtual ~SplineActOperator()
    {
    }

    /** No copies are allowed. */
    SplineActOperator(SplineActOperator const &) = delete;

    /** No assignments are allowed. */
    void operator=(SplineActOperator const &) = delete;

  protected:
    virtual void computeForward(OperatorOutputVector &&outputs,
                                const OperatorInputVector &inputs);

    virtual void computeAdjoint(OperatorOutputVector &&outputs,
                                const OperatorInputVector &inputs);
};

} // namespace optox
