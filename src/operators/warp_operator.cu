///@file warp_operator.cu
///@brief Operator that warps an image given a flow field
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 09.2019


#include "utils.h"
#include "tensor/d_tensor.h"
#include "warp_operator.h"
#include "reduce.cuh"


template <typename T, optox::WarpMode M>
__global__ void warp(
    typename optox::DTensor<T, 4>::Ref out,
    typename optox::DTensor<T, 4>::Ref grad_u,
    const typename optox::DTensor<T, 4>::ConstRef grad_out,
    const typename optox::DTensor<T, 4>::ConstRef u,
    const typename optox::DTensor<T, 4>::ConstRef x,
    int is)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int iz = threadIdx.z + blockIdx.z * blockDim.z;

    if (ix < x.size_[3] && iy < x.size_[2] && iz < x.size_[1])
    {
        // first get the flow
        const T dx = u(is, iy, ix, 0);
        const T dy = u(is, iy, ix, 1);

        // compute the interpolation coefficients
        int ix_f = floorf(ix + dx);
        int iy_f = floorf(iy + dy);

        int ix_c = ix_f + 1;
        int iy_c = iy_f + 1;

        const T w = ix + dx - ix_f;
        const T h = iy + dy - iy_f;

        if (M == optox::WarpMode::replicate)
        {
            ix_f = max(0, min(ix_f, static_cast<int>(x.size_[3] - 1)));
            iy_f = max(0, min(iy_f, static_cast<int>(x.size_[2] - 1)));
            ix_c = max(0, min(ix_c, static_cast<int>(x.size_[3] - 1)));
            iy_c = max(0, min(iy_c, static_cast<int>(x.size_[2] - 1)));
        }

        T i_ff = 0, i_fc = 0;
        if (ix_f >= 0 && ix_f < x.size_[3])
        {
            if (iy_f >= 0 && iy_f < x.size_[2])
            {
                i_ff = grad_out(is, iz, iy_f, ix_f);
            }

            if (iy_c >= 0 && iy_c < x.size_[2])
            {
                i_fc = grad_out(is, iz, iy_c, ix_f);
            }
        }

        T i_cf = 0, i_cc = 0;
        if (ix_c >= 0 && ix_c < x.size_[3])
        {
            if (iy_f >= 0 && iy_f < x.size_[2])
            {
                i_cf = grad_out(is, iz, iy_f, ix_c);
            }


            if (iy_c >= 0 && iy_c < x.size_[2])
            {
                i_cc = grad_out(is, iz, iy_c, ix_c);
            }
                
        }

        // compute the interpolated output
        out(is, iz, iy, ix) = (1 - h) * (1 - w) * i_ff +
                              (1 - h) * w * i_cf +
                              h * (1 - w) * i_fc + 
                              h * w * i_cc;
                              
        T x_val = x(is, iz, iy, ix);
        atomicAdd(&grad_u(is, iy, ix, 0), ((1 - h) * (i_cf - i_ff) + h * (i_cc - i_fc)) * x_val );
        atomicAdd(&grad_u(is, iy, ix, 1), ((1 - w) * (i_fc - i_ff) + w * (i_cc - i_cf)) * x_val );
    }
}


template<typename T>
void optox::WarpOperator<T>::computeForward(optox::OperatorOutputVector &&outputs,
    const optox::OperatorInputVector &inputs)
{
    auto grad_out = this->template getInput<T, 4>(0, inputs);
    auto u = this->template getInput<T, 4>(1, inputs);
    auto x = this->template getInput<T, 4>(2, inputs);
    auto out = this->template getOutput<T, 4>(0, outputs);
    auto grad_u = this->template getOutput<T, 4>(1, outputs);

    if (x->size() != out->size() || 
        x->size()[0] != u->size()[0] || 
        x->size()[2] != u->size()[1] ||
        x->size()[3] != u->size()[2] || 
        u->size()[3] != 2)
        THROW_OPTOXEXCEPTION("WarpOperator: unsupported size");

    dim3 dim_block = dim3(32, 32, 1);
    dim3 dim_grid = dim3(divUp(x->size()[3], dim_block.x),
                         divUp(x->size()[2], dim_block.y),
                         divUp(x->size()[1], dim_block.z));

    // clear the weights gradient
    out->fill(0);
    grad_u->fill(0);

    switch (mode_)
    {
        case optox::WarpMode::replicate:
        {
            for (unsigned int s = 0; s < x->size()[0]; ++s)
            {
                warp<T, optox::WarpMode::replicate> <<<dim_grid, dim_block, 0, this->stream_>>>(*out, *grad_u, *grad_out, *u, *x, s);
            }
            break;
        }
        case optox::WarpMode::zeros:
        {
            for (unsigned int s = 0; s < x->size()[0]; ++s)
            {
                warp<T, optox::WarpMode::zeros> <<<dim_grid, dim_block, 0, this->stream_>>>(*out, *grad_u, *grad_out, *u, *x, s);
            }
            break;
        }
    }
    OPTOX_CUDA_CHECK;
}


template <typename T, optox::WarpMode M>
__global__ void warp_grad(
    typename optox::DTensor<T, 4>::Ref grad_x,
    typename optox::DTensor<T, 4>::Ref grad_u,
    const typename optox::DTensor<T, 4>::ConstRef grad_out,
    const typename optox::DTensor<T, 4>::ConstRef u,
    const typename optox::DTensor<T, 4>::ConstRef x,
    int is)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int iz = threadIdx.z + blockIdx.z * blockDim.z;

    if (ix < grad_x.size_[3] && iy < grad_x.size_[2] && iz < grad_x.size_[1])
    {
        // first get the flow
        const T dx = u(is, iy, ix, 0);
        const T dy = u(is, iy, ix, 1);

        // get the output gradient
        const T grad_val = grad_out(is, iz, iy, ix);

        // compute the interpolation coefficients
        int ix_f = floorf(ix + dx);
        int iy_f = floorf(iy + dy);

        int ix_c = ix_f + 1;
        int iy_c = iy_f + 1;

        const T w = ix + dx - ix_f;
        const T h = iy + dy - iy_f;

        if (M == optox::WarpMode::replicate)
        {
            ix_f = max(0, min(ix_f, static_cast<int>(grad_x.size_[3]-1)));
            iy_f = max(0, min(iy_f, static_cast<int>(grad_x.size_[2]-1)));
            ix_c = max(0, min(ix_c, static_cast<int>(grad_x.size_[3]-1)));
            iy_c = max(0, min(iy_c, static_cast<int>(grad_x.size_[2]-1)));
        }

        T i_ff = 0, i_fc = 0;
        if (ix_f >= 0 && ix_f < grad_x.size_[3])
        {
            if (iy_f >= 0 && iy_f < grad_x.size_[2])
            {
                atomicAdd(&grad_x(is, iz, iy_f, ix_f), (1 - h) * (1 - w) * grad_val);
                i_ff = x(is, iz, iy_f, ix_f);
            }

            if (iy_c >= 0 && iy_c < grad_x.size_[2])
            {
                atomicAdd(&grad_x(is, iz, iy_c, ix_f), h * (1 - w) * grad_val);
                i_fc = x(is, iz, iy_c, ix_f);
            }
        }

        T i_cf = 0, i_cc = 0;
        if (ix_c >= 0 && ix_c < grad_x.size_[3])
        {
            if (iy_f >= 0 && iy_f < grad_x.size_[2])
            {
                atomicAdd(&grad_x(is, iz, iy_f, ix_c), (1 - h) * w * grad_val);
                i_cf = x(is, iz, iy_f, ix_c);
            }

            if (iy_c >= 0 && iy_c < grad_x.size_[2])
            {
                atomicAdd(&grad_x(is, iz, iy_c, ix_c), h * w * grad_val);
                i_cc = x(is, iz, iy_c, ix_c);
            }
        }

        atomicAdd(&grad_u(is, iy, ix, 0), ((1 - h) * (i_cf - i_ff) + h * (i_cc - i_fc)) * grad_val );
        atomicAdd(&grad_u(is, iy, ix, 1), ((1 - w) * (i_fc - i_ff) + w * (i_cc - i_cf)) * grad_val );
    }
}


template<typename T>
void optox::WarpOperator<T>::computeAdjoint(optox::OperatorOutputVector &&outputs,
    const optox::OperatorInputVector &inputs)
{
    auto grad_out = this->template getInput<T, 4>(0, inputs);
    auto u = this->template getInput<T, 4>(1, inputs);
    auto x = this->template getInput<T, 4>(2, inputs);

    auto grad_x = this->template getOutput<T, 4>(0, outputs);
    auto grad_u = this->template getOutput<T, 4>(1, outputs);

    // clear the weights gradient
    grad_x->fill(0);
    grad_u->fill(0);

    if (grad_x->size() != grad_out->size() || 
        grad_x->size()[0] != u->size()[0] || grad_x->size()[2] != u->size()[1] || 
        grad_x->size()[3] != u->size()[2] || u->size()[3] != 2)
        THROW_OPTOXEXCEPTION("WarpOperator: unsupported size");

    dim3 dim_block = dim3(32, 32, 1);
    dim3 dim_grid = dim3(divUp(grad_out->size()[3], dim_block.x),
                         divUp(grad_out->size()[2], dim_block.y),
                         divUp(grad_out->size()[1], dim_block.z));

    switch (mode_)
    {
        case optox::WarpMode::replicate:
        {
            for (unsigned int s = 0; s < grad_out->size()[0]; ++s)
            {
                warp_grad<T, optox::WarpMode::replicate> <<<dim_grid, dim_block, 0, this->stream_>>>(*grad_x, *grad_u, *grad_out, *u, *x, s);
            }
            break;
        }
        case optox::WarpMode::zeros:
        {
            for (unsigned int s = 0; s < grad_out->size()[0]; ++s)
            {
                warp_grad<T, optox::WarpMode::zeros> <<<dim_grid, dim_block, 0, this->stream_>>>(*grad_x, *grad_u, *grad_out, *u, *x, s);
            }
            break;
        }
    }
    OPTOX_CUDA_CHECK;
}

#define REGISTER_OP(T) \
    template class optox::WarpOperator<T>;

OPTOX_CALL_REAL_NUMBER_TYPES(REGISTER_OP);
#undef REGISTER_OP
