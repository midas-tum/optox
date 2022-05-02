///@file pad2d_operator.h
///@brief Operator that pads an image given with symmetric boundary conndition
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 01.202

#include "ioperator.h"

namespace optox
{
/**
 * @enum PaddingMode
 * @brief PaddingMode for border handling
 */
enum class PaddingMode
{
    /** @brief Symmetric padding of border pixels.
     * 
     * 1D Padding of `[1, 2, 3]` by 1px with `symmetric` padding results in `[1, 1, 2, 3, 3]`
     */
    symmetric,
    /** @brief Reflect padding of border pixels. 
     * 
     * 1D Padding of `[1, 2, 3]` by 1px with `symmetric` padding results in `[2, 1, 2, 3, 2]`
    */
    reflect,
    /** @brief Replicate padding of border pixels. */
    replicate
};

template <typename T>
class OPTOX_DLLAPI Pad1dOperator : public IOperator
{
  public:
    /** Special Constructor.
     * 
     * @param left Number of pixels to pad on the left side.
     * @param right Number of pixels to pad on the right side.
     * @param mode Padding mode. Choices are defined via `PaddingMode`
    */
    Pad1dOperator(int left, int right, const std::string &mode) : IOperator(),
        left_(left), right_(right)
    {
        if (mode == "symmetric")
            mode_ = PaddingMode::symmetric;
        else if (mode == "reflect")
            mode_ = PaddingMode::reflect;
        else if (mode == "replicate")
            mode_ = PaddingMode::replicate;
        else
            THROW_OPTOXEXCEPTION("Pad1dOperator: invalid mode!");
    }

    /** Destructor. */
    virtual ~Pad1dOperator()
    {
    }

    /** No copies are allowed. */
    Pad1dOperator(Pad1dOperator const &) = delete;

    /** No assignments are allowed. */
    void operator=(Pad1dOperator const &) = delete;

    /** Compute total padding in x direction. */
    int paddingX() const
    {
        return this->left_ + this->right_;
    }

  protected:
    /** 1D padding of the input image.
     * 
     * - `N` batch size
     * - `W` width of image
     * - `left` padding left
     * - `right` padding right
     * 
     * @param inputs Vector of inputs with length 1
     * - [0] Input image of size `[N, W]`
     * 
     * @param outputs Vector of outputs with length 1
     * - [0] Padded image of size `[N, W+left+right]`
     */
    virtual void computeForward(OperatorOutputVector &&outputs,
                                const OperatorInputVector &inputs);

    /** Transpose (adjoint) 1D padding of the input image.
     * 
     * - `N` batch size
     * - `W` width of image
     * - `left` padding left
     * - `right` padding right
     * 
     * @param inputs Vector of inputs with length 1
     * - [0] Input image of size `[N, W]`
     * 
     * @param outputs Vector of outputs with length 1
     * - [0] Transpose padded image of size `[N, W-left-right]`
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
    int left_;
    int right_;
    PaddingMode mode_;
};

template <typename T>
class OPTOX_DLLAPI Pad2dOperator : public IOperator
{
  public:
    /** Special Constructor.
     * 
     * @param left Number of pixels to pad on the left side.
     * @param right Number of pixels to pad on the right side.
     * @param bottom Number of pixels to pad on the bottom side.
     * @param top Number of pixels to pad on the top side.
     * @param mode Padding mode. Choices are defined via `PaddingMode`
    */
    Pad2dOperator(int left, int right, int top, int bottom, const std::string &mode) : IOperator(),
        left_(left), right_(right), top_(top), bottom_(bottom)
    {
        if (mode == "symmetric")
            mode_ = PaddingMode::symmetric;
        else if (mode == "reflect")
            mode_ = PaddingMode::reflect;
        else if (mode == "replicate")
            mode_ = PaddingMode::replicate;
        else
            THROW_OPTOXEXCEPTION("Pad2dOperator: invalid mode!");
    }

    /** Destructor. */
    virtual ~Pad2dOperator()
    {
    }

    /** No copies are allowed. */
    Pad2dOperator(Pad2dOperator const &) = delete;

    /** No assignments are allowed. */
    void operator=(Pad2dOperator const &) = delete;

    /** Compute total padding in x direction. */
    int paddingX() const
    {
        return this->left_ + this->right_;
    }

    /** Compute total padding in y direction. */
    int paddingY() const
    {
        return this->top_ + this->bottom_;
    }

  protected:
    /** 2D padding of the input image.
     * 
     * - `N` batch size
     * - `H` height of image
     * - `W` width of image
     * - `left` padding left
     * - `right` padding right
     * - `bottom` padding bottom
     * - `top` padding top
     * 
     * @param inputs Vector of inputs with length 1
     * - [0] Input image of size `[N, H, W]`
     * 
     * @param outputs Vector of outputs with length 1
     * - [0] Padded image of size `[N, H+bottoom+top, W+left+right]`
     */
    virtual void computeForward(OperatorOutputVector &&outputs,
                                const OperatorInputVector &inputs);

    /** Transpose (adjoint) 2D padding of the input image.
     * 
     * - `N` batch size
     * - `H` height of image
     * - `W` width of image
     * - `left` padding left
     * - `right` padding right
     * - `bottom` padding bottom
     * - `top` padding 
     * 
     * @param inputs Vector of inputs with length 1
     * - [0] Input image of size `[N, H, W]`
     * 
     * @param outputs Vector of outputs with length 1
     * - [0] Transpose padded image of size `[N, H-bottom-top, W-left-right]`
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
    int left_;
    int right_;
    int top_;
    int bottom_;
    PaddingMode mode_;
};


template <typename T>
class OPTOX_DLLAPI Pad3dOperator : public IOperator
{
  public:
    /** Special Constructor.
     * 
     * @param left Number of pixels to pad on the left side.
     * @param right Number of pixels to pad on the right side.
     * @param bottom Number of pixels to pad on the bottom side.
     * @param top Number of pixels to pad on the top side.
     * @param front Number of pixels to pad on the front side.
     * @param back Number of pixels to pad on the back side.
     * @param mode Padding mode. Choices are defined via `PaddingMode`
    */
    Pad3dOperator(int left, int right, int top, int bottom, int front, int back, const std::string &mode) : IOperator(),
        left_(left), right_(right), top_(top), bottom_(bottom), front_(front), back_(back)
    {
        if (mode == "symmetric") mode_ = PaddingMode::symmetric;
        else if (mode == "reflect") mode_ = PaddingMode::reflect;
        else if (mode == "replicate") mode_ = PaddingMode::replicate;
        else THROW_OPTOXEXCEPTION("Pad3dOperator: invalid mode!");
    }

    /** Destructor. */
    virtual ~Pad3dOperator()
    {
    }

    /** No copies are allowed. */
    Pad3dOperator(Pad3dOperator const &) = delete;

    /** No assignments are allowed. */
    void operator=(Pad3dOperator const &) = delete;

    /** Compute total padding in x direction. */
    int paddingX() const
    {
        return this->left_ + this->right_;
    }

    /** Compute total padding in y direction. */
    int paddingY() const
    {
        return this->top_ + this->bottom_;
    }

    /** Compute total padding in z direction. */
    int paddingZ() const
    {
        return this->front_ + this->back_;
    }

  protected:
    /** 3D padding of the input image.
     * 
     * - `N` batch size
     * - `D` depth of image
     * - `H` height of image
     * - `W` width of image
     * - `left` padding left
     * - `right` padding right
     * - `bottom` padding bottom
     * - `top` padding top
     * - `front` padding front
     * - `back` padding back
     * 
     * @param inputs Vector of inputs with length 1
     * - [0] Input image of size `[N, D, H, W]`
     * 
     * @param outputs Vector of outputs with length 1
     * - [0] Padded image of size `[N, D+front+back, H+bottoom+top, W+left+right]`
     */
    virtual void computeForward(OperatorOutputVector &&outputs,
                                const OperatorInputVector &inputs);

    /** Transpose (adjoint) 3D padding of the input image.
     * 
     * - `N` batch size
     * - `D` depth of image
     * - `H` height of image
     * - `W` width of image
     * - `left` padding left
     * - `right` padding right
     * - `bottom` padding bottom
     * - `top` padding top
     * - `front` padding front
     * - `back` padding back
     * 
     * @param inputs Vector of inputs with length 1
     * - [0] Input image of size `[N, D, H, W]`
     * 
     * @param outputs Vector of outputs with length 1
     * - [0] Transpose padded image of size `[N, D-front-back, H-bottom-top, W-left-right]`
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
    int left_;
    int right_;
    int top_;
    int bottom_;
    int front_;
    int back_;
    PaddingMode mode_;
};


} // namespace optox
