///@file th_gpunufft_operator.cpp
///@brief PyTorch wrappers for gpuNUFFT operator
///@author Kerstin Hammernik <hammernik@icg.tugraz.at>
///@date 08.2019

#include <vector>

#include "th_utils.h"
#include "typetraits.h"
#include "operators/gpunufft_operator.h"

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <c10/cuda/CUDAStream.h>


template<typename T>
at::Tensor forward(optox::GPUNufftOperator<T> &op, at::Tensor th_img, at::Tensor th_csm, at::Tensor th_trajectory, at::Tensor th_dcf)
{
    typedef typename optox::type_trait<T>::complex_type complex_type;
    typedef typename optox::type_trait<T>::real_type real_type;

    // parse the input tensors
    auto img = getComplexDTensorTorch<complex_type, 3>(th_img);
    auto csm = getComplexDTensorTorch<complex_type, 3>(th_csm);
    auto trajectory = getDTensorTorch<real_type, 3>(th_trajectory);
    auto dcf = getDTensorTorch<real_type, 3>(th_dcf);

    // allocate the output tensor
    auto rawdata_shape = th_trajectory.sizes().vec();
    rawdata_shape[1] = th_csm.sizes().vec()[0]; // add number of channels
    rawdata_shape.push_back(2); // make complex
    auto th_rawdata = at::empty(rawdata_shape, th_img.options());
    auto rawdata = getComplexDTensorTorch<complex_type, 3>(th_rawdata);

    op.forward({rawdata.get()}, {img.get(), csm.get(), trajectory.get(), dcf.get()});

    return th_rawdata;
}

template<typename T>
at::Tensor adjoint(optox::GPUNufftOperator<T> &op, at::Tensor th_rawdata, at::Tensor th_csm, at::Tensor th_trajectory, at::Tensor th_dcf)
{
    typedef typename optox::type_trait<T>::complex_type complex_type;
    typedef typename optox::type_trait<T>::real_type real_type;

    // parse the input tensors
    auto rawdata = getComplexDTensorTorch<complex_type, 3>(th_rawdata);
    auto csm = getComplexDTensorTorch<complex_type, 3>(th_csm);
    auto trajectory = getDTensorTorch<real_type, 3>(th_trajectory);
    auto dcf = getDTensorTorch<real_type, 3>(th_dcf);

    // allocate the output tensor
    auto img_shape = th_rawdata.sizes().vec();
    img_shape[1] = op.getImgDim();
    img_shape[2] = op.getImgDim();

    auto th_img = at::empty(img_shape, th_rawdata.options());
    auto img = getComplexDTensorTorch<complex_type, 3>(th_img);
    
    op.adjoint({img.get()}, {rawdata.get(), csm.get(), trajectory.get(), dcf.get()});

    return th_img;
}

template<typename T>
void declare_op(py::module &m, const std::string &typestr)
{
    std::string pyclass_name = std::string("GPUNufft_") + typestr;
    py::class_<optox::GPUNufftOperator<T>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
    .def(py::init<const int&, const T&, const int&, const int&>())
    .def("forward", forward<T>)
    .def("adjoint", adjoint<T>);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    declare_op<float>(m, "float");
}
