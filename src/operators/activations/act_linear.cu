///@file act_linear.cu
///@brief linear interpolation activation function operator
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 01.2019


#include <iu/iucore.h>
#include <iu/iumath.h>
#include <type_traits>

#include "act_linear.h"
#include "utils.cuh"

template<typename T>
__device__ __forceinline__ T baseFunction(T x)
{
    if (x >= -1 && x < 0)
        return x + 1;
    else if (x >= 0 && x < 1)
        return -x + 1;
    else
        return 0;
}

template<typename T>
__device__ __forceinline__ T baseFunctionPrime(T x)
{
    if (x >= -1 && x < 0)
        return 1;
    else if (x >= 0 && x < 1)
        return -1;
    else
        return 0;
}

template<typename T>
__global__ void act_linear_forward_kernel(
    typename iu::LinearDeviceMemory<T, 2>::KernelData output,
    const typename iu::LinearDeviceMemory<T, 2>::KernelData input,
    const typename iu::LinearDeviceMemory<T, 2>::KernelData weights,
    T vmin, T vmax)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= input.size_[0] || y >= input.size_[1])
        return;

    const int Nw = weights.size_[0];

    const T sigma = (vmax - vmin) / (Nw - 1);
    const T k = ((vmax - vmin) / (Nw - 1));

    const T inp_pos = input(x, y);
    T val = 0;
    for (int i = 0; i < Nw; ++i)
    {
        // compute the base function
        const T mu = k * i + vmin;
        const T base_function = baseFunction<T>((inp_pos - mu)/sigma);
        val += weights(i, y) * base_function;
    }

    output(x, y) = val;
}


template<typename T>
__global__ void act_linear_backward_kernel(
    typename iu::LinearDeviceMemory<T, 2>::KernelData grad_input,
    typename iu::LinearDeviceMemory<T, 2>::KernelData grad_weights,
    const typename iu::LinearDeviceMemory<T, 2>::KernelData input,
    const typename iu::LinearDeviceMemory<T, 2>::KernelData weights,
    const typename iu::LinearDeviceMemory<T, 2>::KernelData grad_output,
    T vmin, T vmax)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    extern __shared__ __align__(sizeof(T)) unsigned char sbuffer[];
    T *sdata = reinterpret_cast<T*>(sbuffer);

    if (x >= input.size_[0] || y >= input.size_[1])
    {
        sdata[tid] = 0;
        return;
    }

    const int Nw = weights.size_[0];

    const T sigma = (vmax - vmin) / (Nw - 1);
    const T k = ((vmax - vmin) / (Nw - 1));

    const T inp_pos = input(x, y);
    const T grad_out_pos = grad_output(x, y);
    T grad_inp = 0;
    for (int i = 0; i < Nw; ++i)
    {
        // compute the base function and its derivative
        const T mu = k * i + vmin;
        const T base_function = baseFunction<T>((inp_pos - mu)/sigma);
        const T base_function_prime = baseFunctionPrime<T>((inp_pos - mu)/sigma)/sigma;
        // backpropagate the gradient to the input
        grad_inp += weights(i, y) * base_function_prime * grad_out_pos;

        // backpropagate the gradient to a single weight
        sdata[tid] = base_function * grad_out_pos;

        // parallel reduction along outer dimensions
        __syncthreads();

        parallelReduce(sdata, tid, blockDim.x);

        if(tid == 0)
            atomicAdd(&(grad_weights(i, y)), sdata[tid]);
    }
    grad_input(x, y) = grad_inp;
}


template<typename T>
void optox::LinearActOperator<T>::computeForward(optox::OperatorOutputVector &&outputs,
    const optox::OperatorInputVector &inputs)
{
    auto input = this->template getInput<T, 2>(0, inputs);
    auto weights = this->template getInput<T, 2>(1, inputs);

    auto output = this->template getOutput<T, 2>(0, outputs);

    this->checkSize(input->size(), weights->size());

    int thread_per_block = 256;
    dim3 dim_block = dim3(thread_per_block, 1);
    dim3 block_count = dim3(iu::divUp(input->size()[0], dim_block.x),
                            input->size()[1]);

    act_linear_forward_kernel<T><<<block_count, dim_block, 0, this->stream_>>>(
        *output,
        *input, *weights,
        this->vmin_, this->vmax_);
}

template<typename T>
void optox::LinearActOperator<T>::computeAdjoint(optox::OperatorOutputVector &&outputs,
    const optox::OperatorInputVector &inputs)
{
    auto input = this->template getInput<T, 2>(0, inputs);
    auto weights = this->template getInput<T, 2>(1, inputs);
    auto grad_output = this->template getInput<T, 2>(2, inputs);

    auto grad_input = this->template getOutput<T, 2>(0, outputs);
    auto grad_weights = this->template getOutput<T, 2>(1, outputs);

    this->checkSize(input->size(), weights->size());

    // clear the weights gradient
    iu::math::fill(*grad_weights, static_cast<T>(0));

    int thread_per_block = 256;
    dim3 dim_block = dim3(thread_per_block, 1);
    dim3 block_count = dim3(iu::divUp(input->size()[0], dim_block.x),
                            input->size()[1]);

    act_linear_backward_kernel<T><<<block_count, dim_block, thread_per_block * sizeof(T), this->stream_>>>(
        *grad_input, *grad_weights,
        *input, *weights, *grad_output,
        this->vmin_, this->vmax_);
}


#define REGISTER_OP(T) \
    template class optox::LinearActOperator<T>;

OPTOX_CALL_REAL_NUMBER_TYPES(REGISTER_OP);
#undef REGISTER_OP
