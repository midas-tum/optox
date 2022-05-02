///@file demosaicing_operator.h
///@brief demosaicing operator
///@author Joana Grah <joana.grah@icg.tugraz.at>
///@date 09.07.2018

#include "ioperator.h"

#pragma once

namespace optox
{
/**
 * @enum BayerPattern
 * @brief Bayer Pattern for demosaicing
 */
enum class BayerPattern
{
    RGGB,
    BGGR,
    GBRG,
    GRBG,
};

BayerPattern fromString(const std::string & str)
{
    if (str == "RGGB")
        return BayerPattern::RGGB;
    else if (str == "BGGR")
        return BayerPattern::BGGR;
    else if (str == "GBRG")
        return BayerPattern::GBRG;
    else if (str == "GRBG")
        return BayerPattern::GRBG;
    else 
        throw std::runtime_error("Invalid bayer pattern!");
}

template <typename T>
class OPTOX_DLLAPI DemosaicingOperator : public IOperator
{
  public:
    /** Special Constructor. 
     * 
     * @param pattern Bayer pattern. Choices are defined via `BayerPattern`.
    */
    DemosaicingOperator(const std::string &pattern) : IOperator(), 
        pattern_(fromString(pattern))
    {
    }

    /** Destructor. */
    virtual ~DemosaicingOperator()
    {
    }

    /** No copies are allowed. */
    DemosaicingOperator(DemosaicingOperator const &) = delete;

    /** No assignments are allowed. */
    void operator=(DemosaicingOperator const &) = delete;

  protected:
    /**
     * @brief Demosaicing forward operator
     * 
     * @param inputs Vector of inputs with length 1
     * - [0] Input RBG image of size `[N, H, W, 3]`.
     * @param outputs  Vector of outputs with length 1
     * - [0] Output gray image of size `[N, H, W, 1]`.
     */
    virtual void computeForward(OperatorOutputVector &&outputs,
                                const OperatorInputVector &inputs);

    /**
     * @brief Demosaicing adjoint operator
     * 
     * @param inputs Vector of inputs with length 1
     * - [0] Input gray image of size `[N, H, W, 1]`.
     * @param outputs  Vector of outputs with length 1
     * - [0] Output RGB image of size `[N, H, W, 3]`.
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

private:
    BayerPattern pattern_;
};

} // namespace optox
