///@file nabla_operator.cu
///@brief Operator that computes the forward differences along all dimensions
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 09.07.2018


#include "utils.h"
#include "tensor/d_tensor.h"
#include "nabla_operator.h"

template<typename T>
__global__ void forward_differences(
    typename optox::DTensor<T, 3>::Ref y,
    const typename optox::DTensor<T, 2>::ConstRef x)
{
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < x.size_[1] && iy < x.size_[0])
    {

        const int xp = ix + (ix < x.size_[1] - 1);
        const int yp = iy + (iy < x.size_[0] - 1);

        y(0, iy, ix) = x(iy, xp) - x(iy, ix);
        y(1, iy, ix) = x(yp, ix) - x(iy, ix);
    }
}

template<typename T>
__global__ void forward_differences(
    typename optox::DTensor<T, 4>::Ref y,
    const typename optox::DTensor<T, 3>::ConstRef x)
{
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    int iz = blockDim.z * blockIdx.z + threadIdx.z;
  
    if (ix < x.size_[0] && iy < x.size_[1] && iz < x.size_[2])
    {
        const int xp = ix + (ix < x.size_[0] - 1);
        const int yp = iy + (iy < x.size_[1] - 1);
        const int zp = iz + (iz < x.size_[2] - 1);
        
        y(0, iz, iy , ix) = x(iz, iy, xp) - x(iz, iy, ix);
        y(1, iz, iy , ix) = x(iz, yp, ix) - x(iz, iy, ix);
        y(2, iz, iy , ix) = x(zp, iy, ix) - x(iz, iy, ix);
    }
}

template<typename T, unsigned int N>
void optox::NablaOperator<T, N>::computeForward(optox::OperatorOutputVector &&outputs,
    const optox::OperatorInputVector &inputs)
{
    auto x = this->template getInput<T, N>(0, inputs);
    auto y = this->template getOutput<T, N+1>(0, outputs);

    if (y->size()[0] != N)
        THROW_OPTOXEXCEPTION("NablaOperator: unsupported size");

    dim3 dim_block;
    dim3 dim_grid;
    if (N == 2)
    {
        dim_block = dim3(32, 32);
        dim_grid = dim3(divUp(x->size()[1], dim_block.x),
                        divUp(x->size()[0], dim_block.y));
    }
    else if (N == 3)
    {
        dim_block = dim3(16, 16, 3);
        dim_grid = dim3(divUp(x->size()[2], dim_block.x),
                        divUp(x->size()[1], dim_block.y),
                        divUp(x->size()[0], dim_block.z));
    }
    else
        THROW_OPTOXEXCEPTION("NablaOperator: unsupported dimension");

    forward_differences<T> <<<dim_grid, dim_block, 0, this->stream_>>>(*y, *x);
    OPTOX_CUDA_CHECK;
}


template<typename T>
__global__ void backward_differences(
    typename optox::DTensor<T, 2>::Ref x,
    const typename optox::DTensor<T, 3>::ConstRef y)
{
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < x.size_[1] && iy < x.size_[0])
    {
        T div = (ix > 0) ? 
                        (ix < x.size_[1] - 1) ?
                                            -y(0, iy, ix) + y(0, iy, ix - 1)
                                            :
                                            y(0, iy, ix - 1)
                        :
                        -y(0, iy, ix);

        div += (iy > 0) ? 
                        (iy < x.size_[0] - 1) ?
                                            -y(1, iy, ix) + y(1, iy - 1, ix)
                                            :
                                            y(1, iy - 1, ix)
                        :
                        -y(1, iy, ix);

        x(iy, ix) = div;
    }
}

template<typename T>
__global__ void backward_differences(
    typename optox::DTensor<T, 3>::Ref x,
    const typename optox::DTensor<T, 4>::ConstRef y)
{
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    int iz = blockDim.z * blockIdx.z + threadIdx.z;
  
    if (ix < x.size_[2] && iy < x.size_[1] && iz < x.size_[0])
    {
        T div = (ix > 0) ? 
                        (ix < x.size_[2] - 1) ?
                                            -y(0, iz, iy, ix) + y(0, iz, iy, ix - 1)
                                            :
                                            y(0, iz, iy, ix - 1)
                        :
                        -y(0, iz, iy, ix);

        div += (iy > 0) ? 
                        (iy < x.size_[1] - 1) ?
                                            -y(1, iz, iy, ix) + y(1, iz, iy - 1, ix)
                                            :
                                            y(1, iz, iy - 1, ix)
                        :
                        -y(1, iz, iy, ix);

        div += (iz > 0) ? 
                        (iz < x.size_[0] - 1) ?
                                            -y(2, iz, iy, ix) + y(2, iz - 1, iy, ix)
                                            :
                                            y(2, iz - 1, iy, ix)
                        :
                        -y(2, iz, iy, ix);

        x(iz, iy, ix) = div;
    }
}

template<typename T, unsigned int N>
void optox::NablaOperator<T, N>::computeAdjoint(optox::OperatorOutputVector &&outputs,
    const optox::OperatorInputVector &inputs)
{
    auto y = this->template getInput<T, N+1>(0, inputs);
    auto x = this->template getOutput<T, N>(0, outputs);

    if (y->size()[0] != N)
        THROW_OPTOXEXCEPTION("NablaOperator: unsupported size");

    dim3 dim_block;
    dim3 dim_grid;
    if (N == 2)
    {
        dim_block = dim3(32, 32);
        dim_grid = dim3(divUp(x->size()[1], dim_block.x),
                        divUp(x->size()[0], dim_block.y));
    }
    else if (N == 3)
    {
        dim_block = dim3(16, 16, 3);
        dim_grid = dim3(divUp(x->size()[2], dim_block.x),
                        divUp(x->size()[1], dim_block.y),
                        divUp(x->size()[0], dim_block.z));
    }
    else
        THROW_OPTOXEXCEPTION("NablaOperator: unsupported dimension");

    backward_differences<T> <<<dim_grid, dim_block, 0, this->stream_>>>(*x, *y);
    OPTOX_CUDA_CHECK;
}


#define REGISTER_OP_T(T, N) \
    template class optox::NablaOperator<T, N>;;

#define REGISTER_OP(T) \
    REGISTER_OP_T(T, 2) \
    REGISTER_OP_T(T, 3)

OPTOX_CALL_REAL_NUMBER_TYPES(REGISTER_OP);
#undef REGISTER_OP
#undef REGISTER_OP_T
