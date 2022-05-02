#include "utils.h"
#include "tensor/d_tensor.h"
#include "scale_operator.h"
#include "typetraits.h"

template <typename T>
__global__ void scale_kernel(
    typename optox::DTensor<typename optox::type_trait<T>::real_type, 1>::Ref out,
    const typename optox::DTensor<typename optox::type_trait<T>::real_type, 1>::ConstRef in)
{
    // Get index
    int x = threadIdx.x + blockIdx.x * blockDim.x;

    // Use constant scale for now
    T scale = optox::type_trait<T>::make(2.0);

    // Check if indices are in range
    if (x < out.size_[0])
    {
        // Compute the corresponding index 
        out(x) = in(x) * scale;
    }
}

template <typename T>
__global__ void scale_kernel(
    typename optox::DTensor<typename optox::type_trait<T>::complex_type, 1>::Ref out,
    const typename optox::DTensor<typename optox::type_trait<T>::complex_type, 1>::ConstRef in)
{
    // Get index
    int x = threadIdx.x + blockIdx.x * blockDim.x;

    // Use constant scale for now
    T scale = optox::type_trait<T>::make_complex(2.0, 3.0);

    // Check if indices are in range
    if (x < out.size_[0])
    {
        // Compute the corresponding index 
        out(x) = optox::complex_multiply<T>(in(x), scale);
    }
}

template<typename T>
void optox::ScaleOperator<T>::computeForward(optox::OperatorOutputVector &&outputs,
    const optox::OperatorInputVector &inputs)
{
    // Extract input
    auto x = this->template getInput<T, 1>(0, inputs);
    // Extract output
    auto out = this->template getOutput<T, 1>(0, outputs);

    // Check sizes
    if (x->size() != out->size())
        THROW_OPTOXEXCEPTION("ScaleOperator: unsupported size");

    // Kernel specifications
    dim3 dim_block = dim3(128, 1, 1);
    dim3 dim_grid = dim3(divUp(out->size()[1], dim_block.x),
                         1,
                         1);

    // Clear the output
    out->fill(optox::type_trait<T>::make(0));

    // Launch CUDA kernel
    scale_kernel<T> <<<dim_grid, dim_block, 0, this->stream_>>>(*out, *x);
    OPTOX_CUDA_CHECK;
}

template<typename T>
void optox::ScaleOperator<T>::computeAdjoint(optox::OperatorOutputVector &&outputs,
    const optox::OperatorInputVector &inputs)
{
    auto x = this->template getInput<T, 1>(0, inputs);
    auto out = this->template getOutput<T, 1>(0, outputs);

    if (x->size() != out->size())
        THROW_OPTOXEXCEPTION("ScaleOperator: unsupported size");

    dim3 dim_block = dim3(128, 1, 1);
    dim3 dim_grid = dim3(divUp(out->size()[1], dim_block.x),
                         1,
                         1);

    // clear the output
    out->fill(optox::type_trait<T>::make(0));

    scale_kernel<T> <<<dim_grid, dim_block, 0, this->stream_>>>(*out, *x);
    OPTOX_CUDA_CHECK;
}
 
#define REGISTER_OP(T) \
    template class optox::ScaleOperator<T>;

OPTOX_CALL_NUMBER_TYPES(REGISTER_OP);
#undef REGISTER_OP