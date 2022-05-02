///@file warp_operator.h
///@brief Operator that warps an image given a flow field
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 09.2019

#include "ioperator.h"

namespace optox
{
/**
 * @enum WarpMode
 * @brief WarpMode for border handling
 */
enum class WarpMode {
    /** Replicate border pixels. */
    replicate,
    /** Use zeros for pixels outside the image. */
    zeros
    };

/**
 * @class WarpOperator
 * @brief Warps an input with a given flow field.
 */
template <typename T>
class OPTOX_DLLAPI WarpOperator : public IOperator
{
  public:
    /** Special Constructor. 
     * 
     * @param mode Padding mode. Choices are defined via `WarpMode`
    */
    WarpOperator(const std::string &mode) : IOperator()
    {
        if (mode == "replicate") mode_ = WarpMode::replicate;
        else if (mode == "zeros") mode_ = WarpMode::zeros;
        else THROW_OPTOXEXCEPTION("WarpOperator: invalid mode!");
    }

    /** Destructor. */
    virtual ~WarpOperator()
    {
    }

    /** No copies are allowed. */
    WarpOperator(WarpOperator const &) = delete;

    /** No assignments are allowed. */
    void operator=(WarpOperator const &) = delete;

  protected:
    /**
     * @brief Compute forward warping for 2D data.
     * 
     * \remark 3 inputs are required for the forward path. If only the forward path is needed,
     * the input flow field and input image are enough, hence, `inputs[0]` equals to the input image.
     * The third input is only required if the forward function is used for the computation of
     * the gradient of the adjoint operation.
     *  
     * - `batch_size`: batch size
     * - `features`: number of features in the image.
     * - `H`: height of the image.
     * - `W`: width of the image
     * 
     * @param outputs Vector of outputs with length 2
     * - [0] gradient wrt. input `[batch_size, features, H, W]`
     * - [1] gradient wrt. u of size `[batch_size, H, W, 2]`
     * @param inputs Vector of inputs with length 3
     * - [0] gradient wrt. output `[batch_size, features, H, W]`
     * - [1] flow field of size `[batch_size, H, W, 2]`
     * - [2] image of size `[batch_size, features, H, W]`
     */
    virtual void computeForward(OperatorOutputVector &&outputs,
                                const OperatorInputVector &inputs);

    /**
     * @brief Compute adjoint warping for 2D data.
     * 
     * \remark 3 inputs are required for the adjoint path. If only the adjoint path is needed,
     * the input flow field and input image are enough, hence, `inputs[0]` equals to the input image.
     * The third input is only required if the adjoint function is used for the computation of
     * the gradient of the forward operation.
     * 
     * - `batch_size`: batch size
     * - `features`: number of features in the image.
     * - `H`: height of the image.
     * - `W`: width of the image
     * 
     * @param outputs Vector of outputs with length 2
     * - [0] warped mage of size `[batch_size, features, H, W]`
     * - [1] gradient wrt. u of size `[batch_size, H, W, 2]`
     * @param inputs Vector of inputs with length 3
     * - [0] gradient wrt. output `[batch_size, features, H, W]`
     * - [1] flow field of size `[batch_size, H, W, 2]`
     * - [2] image of size `[batch_size, features, H, W]`
     */
    virtual void computeAdjoint(OperatorOutputVector &&outputs,
                                const OperatorInputVector &inputs);

    /** Number of required outputs for the forward operator.
     * @return Number of outputs for forward operator (2)
    */
    virtual unsigned int getNumOutputsForward()
    {
        return 2;
    }

    /** Number of required inputs for the forward operator.
     * @return Number of inputs for forward operator (3)
    */
    virtual unsigned int getNumInputsForward()
    {
        return 3;
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

    private:
        WarpMode mode_;
};

} // namespace optox
