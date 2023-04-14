///@file gpunufft_operator.cu
///@brief wrapper for gpunufft operator
///@author Kerstin Hammernik <k.hammernik@imperial.ac.uk>
///@date 08.2018


#include "utils.h"
#include "tensor/d_tensor.h"
#include "gpunufft_operator.h"
#include "typetraits.h"

#include <gpuNUFFT_operator_factory.hpp>

template<typename T>
void optox::GPUNufftOperator<T>::computeForward(optox::OperatorOutputVector &&outputs,
    const optox::OperatorInputVector &inputs)
{
    typedef typename optox::type_trait<T>::complex_type complex_type;
    typedef typename optox::type_trait<T>::real_type real_type;

    auto img = this->template getInput<complex_type, 3>(0, inputs);
    auto csm = this->template getInput<complex_type, 3>(1, inputs);
    auto trajectory = this->template getInput<real_type, 3>(2, inputs);
    auto dcf = this->template getInput<real_type, 3>(3, inputs);

    auto rawdata = this->template getOutput<complex_type, 3>(0, outputs);
    rawdata->fill(optox::type_trait<T>::make_complex(0));

    if (rawdata->size()[1] != csm->size()[0])
        THROW_OPTOXEXCEPTION("GPUNufftOperator: rawdata and csm must have same number of channels!");
    if (rawdata->size()[2] != trajectory->size()[2] || rawdata->size()[2] != dcf->size()[2])
        THROW_OPTOXEXCEPTION("GPUNufftOperator: rawdata, dcf and trajectory must have same number of samples!");
    if (csm->size()[1] != img_dim_ || csm->size()[2] != img_dim_ || 
        img->size()[1] != img_dim_ || img->size()[2] != img_dim_)
        THROW_OPTOXEXCEPTION("GPUNufftOperator: image and csm must have img_dim specified in NUFFT config!");

    int samples = dcf->size()[0];

    gpuNUFFT::Dimensions gpunufft_img_dims;
    gpunufft_img_dims.width = img_dim_;
    gpunufft_img_dims.height = img_dim_;
    gpunufft_img_dims.depth = 0;

    gpuNUFFT::GpuNUFFTOperatorFactory factory(true,true,true,false,this->stream_);

    gpuNUFFT::Array<real_type> gpunufft_trajectory;
    gpunufft_trajectory.dim.length = dcf->size()[2];
    int trajectory_offset = trajectory->numel() / samples;

    gpuNUFFT::Array<real_type> gpunufft_dcf;
    gpunufft_dcf.dim.length = dcf->size()[2];
    int dcf_offset = dcf->numel() / samples;

    gpuNUFFT::Array<complex_type> gpunufft_csm;
    gpunufft_csm.dim = gpunufft_img_dims;
    gpunufft_csm.dim.channels = csm->size()[0];
    int csm_offset = 0; //csm->numel() / samples;

    gpuNUFFT::Array<complex_type> gpunufft_img;
    gpunufft_img.dim = gpunufft_img_dims;
    int img_offset = img->numel() / samples;

    gpuNUFFT::Array<complex_type> gpunufft_rawdata;
    gpunufft_rawdata.dim.length = rawdata->size()[2];
    gpunufft_rawdata.dim.channels = rawdata->size()[1];
    int rawdata_offset = rawdata->numel() / samples;

    for (int n = 0; n < samples; n++)
    {
        gpunufft_trajectory.data = const_cast<real_type*>(reinterpret_cast<const real_type*>(trajectory->constptr(n * trajectory_offset)));
        gpunufft_dcf.data = const_cast<real_type*>(reinterpret_cast<const real_type*>(dcf->constptr(n * dcf_offset)));
        gpunufft_csm.data = const_cast<complex_type*>(reinterpret_cast<const complex_type*>(csm->constptr(n * csm_offset)));

        gpuNUFFT::GpuNUFFTOperator* nufft_op = factory.createGpuNUFFTOperator(
                                                gpunufft_trajectory,
                                                gpunufft_dcf, 
                                                gpunufft_csm,
                                                kernel_width_,
                                                sector_width_,
                                                osf_,
                                                gpunufft_img_dims);
        
        gpunufft_img.data = const_cast<complex_type*>(reinterpret_cast<const complex_type*>(img->constptr(n * img_offset)));
        gpunufft_rawdata.data = rawdata->ptr(n * rawdata_offset);

        nufft_op->performForwardGpuNUFFT(gpunufft_img, gpunufft_rawdata);

        delete nufft_op;
    }
}

template<typename T>
void optox::GPUNufftOperator<T>::computeAdjoint(optox::OperatorOutputVector &&outputs,
    const optox::OperatorInputVector &inputs)
{
    typedef typename optox::type_trait<T>::complex_type complex_type;
    typedef typename optox::type_trait<T>::real_type real_type;

    auto rawdata = this->template getInput<complex_type, 3>(0, inputs);
    auto csm = this->template getInput<complex_type, 3>(1, inputs);
    auto trajectory = this->template getInput<real_type, 3>(2, inputs);
    auto dcf = this->template getInput<real_type, 3>(3, inputs);

    auto img = this->template getOutput<complex_type, 3>(0, outputs);
    img->fill(optox::type_trait<T>::make_complex(0));

    if (rawdata->size()[1] != csm->size()[0])
        THROW_OPTOXEXCEPTION("GPUNufftOperator: rawdata and csm must have same number of channels!");
    if (rawdata->size()[2] != trajectory->size()[2] || rawdata->size()[2] != dcf->size()[2])
        THROW_OPTOXEXCEPTION("GPUNufftOperator: rawdata, dcf and trajectory must have same number of samples!");
    if (csm->size()[1] != img_dim_ || csm->size()[2] != img_dim_ || 
        img->size()[1] != img_dim_ || img->size()[2] != img_dim_)
        THROW_OPTOXEXCEPTION("GPUNufftOperator: image and csm must have img_dim specified in NUFFT config!");

    int samples = dcf->size()[0];

    gpuNUFFT::Dimensions gpunufft_img_dims;
    gpunufft_img_dims.width = img_dim_;
    gpunufft_img_dims.height = img_dim_;
    gpunufft_img_dims.depth = 0;

    gpuNUFFT::GpuNUFFTOperatorFactory factory(true,true,true,false,this->stream_);

    gpuNUFFT::Array<real_type> gpunufft_trajectory;
    gpunufft_trajectory.dim.length = dcf->size()[2];
    int trajectory_offset = trajectory->numel() / samples;

    gpuNUFFT::Array<real_type> gpunufft_dcf;
    gpunufft_dcf.dim.length = dcf->size()[2];
    int dcf_offset = dcf->numel() / samples;

    gpuNUFFT::Array<complex_type> gpunufft_csm;
    gpunufft_csm.dim = gpunufft_img_dims;
    gpunufft_csm.dim.channels = csm->size()[0];
    int csm_offset = 0; // csm->numel() / samples;

    gpuNUFFT::Array<complex_type> gpunufft_img;
    gpunufft_img.dim = gpunufft_img_dims;
    int img_offset = img->numel() / samples;

    gpuNUFFT::Array<complex_type> gpunufft_rawdata;
    gpunufft_rawdata.dim.length = rawdata->size()[2];
    gpunufft_rawdata.dim.channels = rawdata->size()[1];
    int rawdata_offset = rawdata->numel() / samples;

    for (int n = 0; n < samples; n++)
    {
        gpunufft_trajectory.data = const_cast<real_type*>(reinterpret_cast<const real_type*>(trajectory->constptr(n * trajectory_offset)));
        gpunufft_dcf.data = const_cast<real_type*>(reinterpret_cast<const real_type*>(dcf->constptr(n * dcf_offset)));
        gpunufft_csm.data = const_cast<complex_type*>(reinterpret_cast<const complex_type*>(csm->constptr(n * csm_offset)));

        gpuNUFFT::GpuNUFFTOperator* nufft_op = factory.createGpuNUFFTOperator(
                                                gpunufft_trajectory,
                                                gpunufft_dcf, 
                                                gpunufft_csm,
                                                kernel_width_,
                                                sector_width_,
                                                osf_,
                                                gpunufft_img_dims);
        
        gpunufft_rawdata.data = const_cast<complex_type*>(reinterpret_cast<const complex_type*>(rawdata->constptr(n * rawdata_offset)));
        gpunufft_img.data = img->ptr(n * img_offset);

        nufft_op->performGpuNUFFTAdj(gpunufft_rawdata, gpunufft_img);

        delete nufft_op;
    }
}

template<typename T>
void optox::GPUNufftSingleCoilOperator<T>::computeForward(optox::OperatorOutputVector &&outputs,
    const optox::OperatorInputVector &inputs)
{
    typedef typename optox::type_trait<T>::complex_type complex_type;
    typedef typename optox::type_trait<T>::real_type real_type;

    auto img = this->template getInput<complex_type, 3>(0, inputs);
    auto trajectory = this->template getInput<real_type, 3>(1, inputs);
    auto dcf = this->template getInput<real_type, 3>(2, inputs);

    auto rawdata = this->template getOutput<complex_type, 3>(0, outputs);


    if (rawdata->size()[2] != trajectory->size()[2] || rawdata->size()[2] != dcf->size()[2])
        THROW_OPTOXEXCEPTION("GPUNufftSingleCoilOperator: rawdata, dcf and trajectory must have same number of samples!");
    if (img->size()[1] != img_dim_ || img->size()[2] != img_dim_)
        THROW_OPTOXEXCEPTION("GPUNufftSingleCoilOperator: image must have img_dim specified in NUFFT config!");

    int samples = dcf->size()[0];

    gpuNUFFT::Dimensions gpunufft_img_dims;
    gpunufft_img_dims.width = img_dim_;
    gpunufft_img_dims.height = img_dim_;
    gpunufft_img_dims.depth = 0;

    gpuNUFFT::GpuNUFFTOperatorFactory factory(true,true,true,false,this->stream_);

    gpuNUFFT::Array<real_type> gpunufft_trajectory;
    gpunufft_trajectory.dim.length = dcf->size()[2];
    int trajectory_offset = trajectory->numel() / samples;

    gpuNUFFT::Array<real_type> gpunufft_dcf;
    gpunufft_dcf.dim.length = dcf->size()[2];
    int dcf_offset = dcf->numel() / samples;

    gpuNUFFT::Array<complex_type> gpunufft_img;
    gpunufft_img.dim = gpunufft_img_dims;
    int img_offset = img->numel() / samples;

    gpuNUFFT::Array<complex_type> gpunufft_rawdata;
    gpunufft_rawdata.dim.length = rawdata->size()[2];
    gpunufft_rawdata.dim.channels = 1;
    int rawdata_offset = rawdata->numel() / samples;

    for (int n = 0; n < samples; n++)
    {
        gpunufft_trajectory.data = const_cast<real_type*>(reinterpret_cast<const real_type*>(trajectory->constptr(n * trajectory_offset)));
        gpunufft_dcf.data = const_cast<real_type*>(reinterpret_cast<const real_type*>(dcf->constptr(n * dcf_offset)));

        gpuNUFFT::GpuNUFFTOperator* nufft_op = factory.createGpuNUFFTOperator(
                                                gpunufft_trajectory,
                                                gpunufft_dcf, 
                                                kernel_width_,
                                                sector_width_,
                                                osf_,
                                                gpunufft_img_dims);
        
        gpunufft_img.data = const_cast<complex_type*>(reinterpret_cast<const complex_type*>(img->constptr(n * img_offset)));
        gpunufft_rawdata.data = rawdata->ptr(n * rawdata_offset);

        nufft_op->performForwardGpuNUFFT(gpunufft_img, gpunufft_rawdata);

        delete nufft_op;
    }
}

template<typename T>
void optox::GPUNufftSingleCoilOperator<T>::computeAdjoint(optox::OperatorOutputVector &&outputs,
    const optox::OperatorInputVector &inputs)
{
    typedef typename optox::type_trait<T>::complex_type complex_type;
    typedef typename optox::type_trait<T>::real_type real_type;

    auto rawdata = this->template getInput<complex_type, 3>(0, inputs);
    auto trajectory = this->template getInput<real_type, 3>(1, inputs);
    auto dcf = this->template getInput<real_type, 3>(2, inputs);

    auto img = this->template getOutput<complex_type, 3>(0, outputs);

    if (rawdata->size()[1] != 1)
        THROW_OPTOXEXCEPTION("GPUNufftSingleCoilOperator: rawdata must be single-coil");
    if (rawdata->size()[2] != trajectory->size()[2] || rawdata->size()[2] != dcf->size()[2])
        THROW_OPTOXEXCEPTION("GPUNufftSingleCoilOperator: rawdata, dcf and trajectory must have same number of samples!");
    if (img->size()[1] != img_dim_ || img->size()[2] != img_dim_)
        THROW_OPTOXEXCEPTION("GPUNufftSingleCoilOperator: image have img_dim specified in NUFFT config!");

    int samples = dcf->size()[0];

    gpuNUFFT::Dimensions gpunufft_img_dims;
    gpunufft_img_dims.width = img_dim_;
    gpunufft_img_dims.height = img_dim_;
    gpunufft_img_dims.depth = 0;

    gpuNUFFT::GpuNUFFTOperatorFactory factory(true,true,true,false,this->stream_);

    gpuNUFFT::Array<real_type> gpunufft_trajectory;
    gpunufft_trajectory.dim.length = dcf->size()[2];
    int trajectory_offset = trajectory->numel() / samples;

    gpuNUFFT::Array<real_type> gpunufft_dcf;
    gpunufft_dcf.dim.length = dcf->size()[2];
    int dcf_offset = dcf->numel() / samples;

    gpuNUFFT::Array<complex_type> gpunufft_img;
    gpunufft_img.dim = gpunufft_img_dims;
    int img_offset = img->numel() / samples;

    gpuNUFFT::Array<complex_type> gpunufft_rawdata;
    gpunufft_rawdata.dim.length = rawdata->size()[2];
    gpunufft_rawdata.dim.channels = 1;
    int rawdata_offset = rawdata->numel() / samples;

    for (int n = 0; n < samples; n++)
    {
        gpunufft_trajectory.data = const_cast<real_type*>(reinterpret_cast<const real_type*>(trajectory->constptr(n * trajectory_offset)));
        gpunufft_dcf.data = const_cast<real_type*>(reinterpret_cast<const real_type*>(dcf->constptr(n * dcf_offset)));

        gpuNUFFT::GpuNUFFTOperator* nufft_op = factory.createGpuNUFFTOperator(
                                                gpunufft_trajectory,
                                                gpunufft_dcf, 
                                                kernel_width_,
                                                sector_width_,
                                                osf_,
                                                gpunufft_img_dims);
        
        gpunufft_rawdata.data = const_cast<complex_type*>(reinterpret_cast<const complex_type*>(rawdata->constptr(n * rawdata_offset)));
        gpunufft_img.data = img->ptr(n * img_offset);

        nufft_op->performGpuNUFFTAdj(gpunufft_rawdata, gpunufft_img);

        delete nufft_op;
    }
}


template class optox::GPUNufftOperator<float>;
template class optox::GPUNufftSingleCoilOperator<float>;