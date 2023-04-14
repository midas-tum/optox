///@file gpunufft_operator.h
///@brief wrapper for gpunufft operator
///@author Kerstin Hammernik  <k.hammernik@imperial.ac.uk>
///@date 08.2018

#include "ioperator.h"

#pragma once

namespace optox
{
/**
 * @class GPUNufftOperator
 * @brief Wrapper for multicoil GpuNUFFT operator
 * 
 * Requires <a href="https://github.com/khammernik/gpuNUFFT">gpuNUFFT</a> with Cuda Streams.
 * Please check out the branch `cuda_streams`.
 */
template <typename T>
class OPTOX_DLLAPI GPUNufftOperator : public IOperator
{
  public:
    /** Special Constructor. 
     * 
     * @param img_dim Image dimensions.
     * @param osf Oversampling factor.
     * @param kernel_width Width of interpolation kernel.
     * @param sector_width Sector size used to split the grid into sectors, which are processed in parallel.
    */
    GPUNufftOperator(const int& img_dim, const T& osf, const int& kernel_width, const int& sector_width) : IOperator(), 
        img_dim_(img_dim), osf_(osf), kernel_width_(kernel_width), sector_width_(sector_width)
    {
    }

    /** Destructor. */
    virtual ~GPUNufftOperator()
    {
    }

    /** No copies are allowed. */
    GPUNufftOperator(GPUNufftOperator const &) = delete;

    /** No assignments are allowed. */
    void operator=(GPUNufftOperator const &) = delete;

    /**
     * @brief Get the image dimension.
     * 
     * We are reconstructing squared images, therefore, we only return an `int` here.
     * 
     * @return Image dimension
     */
    int getImgDim() const
    {
        return img_dim_;
    }

    /**
     * @brief Get the oversampling factor (osf)
     * 
     * Oversampling in k-space.
     * 
     * @return Oversampling factor
     */
    T getOsf() const
    {
        return osf_;
    }

    /**
     * @brief Get the width of the interpolation kernel.
     * 
     * @return Width of interpolation kernel.
     */
    int getKernelWidth() const
    {
        return kernel_width_;
    }

    /**
     * @brief Get the size of sector
     * 
     * @return Sector size used to split the grid into sectors, which are processed in parallel.
     */
    int getSectorWidth() const
    {
        return sector_width_;
    }

  protected:
    /**
     * @brief Compute forward gpuNUFFT for 2D data
     * 
     * - `samples`: number of frames for dynamic data, set to 1 for static data
     * - `coils`: number of receive coils
     * - `nFE`: number of frequency encoding points
     * - `nSpokes`: number of spokes
     * 
     * @param outputs Vector of outputs with length 1
     * - [0] Rawdata (k-space) of size `[samples, coils, nFE * nSpokes]`
     * @param inputs Vector of inputs with length 4
     * - [0] Image of size `[samples, img_dim_, img_dim_]`
     * - [1] Coil sensitivity maps of size `[coils, img_dim_, img_dim_]`
     * - [2] Sampling trajectory of size `[samples, 2, nFE * nSpokes]`
     * - [3] Density compensation function of size `[samples, 1, nFE * nSpokes]`
     */
    virtual void computeForward(OperatorOutputVector &&outputs,
                                const OperatorInputVector &inputs);

    /**
     * @brief Compute adjoint gpuNUFFT for 2D data
     * 
     * - `samples`: number of frames for dynamic data, set to 1 for static data
     * - `coils`: number of receive coils
     * - `nFE`: number of frequency encoding points
     * - `nSpokes`: number of spokes
     * 
     * @param outputs Vector of outputs with length 1
     * - [0] Image of size `[samples, img_dim_, img_dim_]`
     * @param inputs Vector of inputs with length 4
     * - [0] Rawdata (k-space) of size `[samples, coils, nFE * nSpokes]`
     * - [1] Coil sensitivity maps of size `[coils, img_dim_, img_dim_]`
     * - [2] Sampling trajectory of size `[samples, 2, nFE * nSpokes]`
     * - [3] Density compensation function of size `[samples, 1, nFE * nSpokes]`
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
     * @return Number of inputs for forward operator (4)
    */
    virtual unsigned int getNumInputsForward()
    {
        return 4;
    }

    /** Number of required outputs for the adjoint operator.
     * @return Number of outputs for adjoint operator (1)
    */
    virtual unsigned int getNumOutputsAdjoint()
    {
        return 1;
    }

    /** Number of required inputs for the adjoint operator.
     * @return Number of inputs for adjoint operator (4)
    */
    virtual unsigned int getNumInputsAdjoint()
    {
        return 4;
    }

private:
    /** Size of reconstructed image */
    int img_dim_;
    /** Oversampling factor */
    T osf_;
    /** Width of interpolation kernel */
    int kernel_width_;
    /** Sector size used to split the grid into sectors, which are processed in parallel. */
    int sector_width_;
};


/**
 * @class GPUNufftSingleCoilOperator
 * @brief Wrapper for singlecoil GpuNUFFT operator
 * 
 * Requires <a href="https://github.com/khammernik/gpuNUFFT">gpuNUFFT</a> with Cuda Streams.
 * Please check out the branch `cuda_streams`.
 */
template <typename T>
class OPTOX_DLLAPI GPUNufftSingleCoilOperator : public IOperator
{
  public:
    /** Special Constructor. 
     * 
     * @param img_dim Image dimensions.
     * @param osf Oversampling factor.
     * @param kernel_width Width of interpolation kernel.
     * @param sector_width Sector size used to split the grid into sectors, which are processed in parallel.
    */    GPUNufftSingleCoilOperator(const int& img_dim, const T& osf, const int& kernel_width, const int& sector_width) : IOperator(), 
        img_dim_(img_dim), osf_(osf), kernel_width_(kernel_width), sector_width_(sector_width)
    {
    }

    /** Destructor. */
    virtual ~GPUNufftSingleCoilOperator()
    {
    }

    /** No copies are allowed. */   
    GPUNufftSingleCoilOperator(GPUNufftSingleCoilOperator const &) = delete;
    
    /** No assignments are allowed. */
    void operator=(GPUNufftSingleCoilOperator const &) = delete;

    /**
     * @brief Get the image dimension.
     * 
     * We are reconstructing squared images, therefore, we only return an `int` here.
     * 
     * @return Image dimension
     */
    int getImgDim() const
    {
        return img_dim_;
    }

    /**
     * @brief Get the oversampling factor (osf)
     * 
     * Oversampling in k-space.
     * 
     * @return Oversampling factor
     */
    T getOsf() const
    {
        return osf_;
    }

    /**
     * @brief Get the width of the interpolation kernel.
     * 
     * @return Width of interpolation kernel.
     */
    int getKernelWidth() const
    {
        return kernel_width_;
    }

    /**
     * @brief Get the size of sector
     * 
     * @return Sector size used to split the grid into sectors, which are processed in parallel.
     */
    int getSectorWidth() const
    {
        return sector_width_;
    }

  protected:
    /**
     * @brief Compute forward gpuNUFFT for 2D data
     * 
     * - `samples`: number of frames for dynamic data, set to 1 for static data
     * - `nFE`: number of frequency encoding points
     * - `nSpokes`: number of spokes
     * 
     * @param outputs Vector of outputs with length 1
     * - [0] Rawdata (k-space) of size `[samples, 1, nFE * nSpokes]`
     * @param inputs Vector of inputs with length 3
     * - [0] Image of size `[samples, img_dim_, img_dim_]`
     * - [1] Sampling trajectory of size `[samples, 2, nFE * nSpokes]`
     * - [2] Density compensation function of size `[samples, 1, nFE * nSpokes]`
     */
    virtual void computeForward(OperatorOutputVector &&outputs,
                                const OperatorInputVector &inputs);

    /**
     * @brief Compute adjoint gpuNUFFT for 2D data
     * 
     * - `samples`: number of frames for dynamic data, set to 1 for static data
     * - `nFE`: number of frequency encoding points
     * - `nSpokes`: number of spokes
     * 
     * @param outputs Vector of outputs with length 1
     * - [0] Image of size `[samples, img_dim_, img_dim_]`
     * @param inputs Vector of inputs with length 3
     * - [0] Rawdata (k-space) of size `[samples, 1, nFE * nSpokes]`
     * - [1] Sampling trajectory of size `[samples, 2, nFE * nSpokes]`
     * - [2] Density compensation function of size `[samples, 1, nFE * nSpokes]`
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
     * @return Number of inputs for forward operator (3)
    */
    virtual unsigned int getNumInputsForward()
    {
        return 3;
    }

    /** Number of required outputs for the adjoint operator.
     * @return Number of outputs for adjoint operator (1)
    */
    virtual unsigned int getNumOutputsAdjoint()
    {
        return 1;
    }

    /** Number of required inputs for the adjoint operator.
     * @return Number of inputs for adjoint operator (3)
    */
    virtual unsigned int getNumInputsAdjoint()
    {
        return 3;
    }

  private:
    /** Size of reconstructed image */
    int img_dim_;
    /** Oversampling factor */
    T osf_;
    /** Width of interpolation kernel */
    int kernel_width_;
    /** Sector size used to split the grid into sectors, which are processed in parallel. */
    int sector_width_;
};
} // namespace optox
