///@file py_gpunufft_singlecoil_operator.cpp
///@brief Python wrappers for singlecoil gpuNUFFT operator
///@author Kerstin Hammernik <k.hammernik@imperial.ac.uk>
///@date 01.2020

#include <vector>

#include "py_utils.h"
#include "typetraits.h"
#include "operators/gpunufft_operator.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

template<typename T>
py::array forward(optox::GPUNufftSingleCoilOperator<T> &op, py::array np_img, py::array np_trajectory, py::array np_dcf)
{
    typedef typename optox::type_trait<T>::complex_type complex_type;
    typedef typename optox::type_trait<T>::real_type real_type;

    // parse the input tensors
    auto img = getComplexDTensorNp<complex_type, 3>(np_img);
    auto trajectory = getDTensorNp<real_type, 3>(np_trajectory);
    auto dcf = getDTensorNp<real_type, 3>(np_dcf);

    // allocate the output tensor
    optox::Shape<3> rawdata_shape;
    rawdata_shape[0] = trajectory->size()[0];
    rawdata_shape[1] = 1; // add number of channels
    rawdata_shape[2] = trajectory->size()[2];

    optox::DTensor<complex_type, 3> rawdata(rawdata_shape);

    op.forward({&rawdata}, {img.get(), trajectory.get(), dcf.get()});
    return dComplexTensorToNp(rawdata);
}

template<typename T>
py::array adjoint(optox::GPUNufftSingleCoilOperator<T> &op, py::array np_rawdata, py::array np_trajectory, py::array np_dcf)
{
    typedef typename optox::type_trait<T>::complex_type complex_type;
    typedef typename optox::type_trait<T>::real_type real_type;

    // parse the input tensors
    auto rawdata = getComplexDTensorNp<complex_type, 3>(np_rawdata);
    auto trajectory = getDTensorNp<real_type, 3>(np_trajectory);
    auto dcf = getDTensorNp<real_type, 3>(np_dcf);

    // allocate the output tensor
    optox::Shape<3> img_shape;
    img_shape[0] = rawdata->size()[0]; 
    img_shape[1] = op.getImgDim();
    img_shape[2] = op.getImgDim();
   
    optox::DTensor<complex_type, 3> img(img_shape);
   
    op.adjoint({&img}, {rawdata.get(), trajectory.get(), dcf.get()});

    return dComplexTensorToNp(img);
}

template<typename T>
void declare_op(py::module &m, const std::string &typestr)
{
    std::string pyclass_name = std::string("GPUNufft_singlecoil_") + typestr;
    py::class_<optox::GPUNufftSingleCoilOperator<T>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
    .def(py::init<const int&, const T&, const int&, const int&>())
    .def("forward", forward<T>)
    .def("adjoint", adjoint<T>)
    // add pickle support
    .def("__getstate__", [](const optox::GPUNufftSingleCoilOperator<T> &op) {
        /* Return a tuple that fully encodes the state of the object */
        return py::make_tuple(op.getImgDim(), op.getOsf(), op.getKernelWidth(), op.getSectorWidth());
    })
    .def("__setstate__", [](optox::GPUNufftSingleCoilOperator<T> &op, py::tuple t) {
        if (t.size() != 4)
            throw std::runtime_error("Invalid state!");
        /* Invoke the constructor (need to use in-place version) */
        new (&op) optox::GPUNufftSingleCoilOperator<T>(t[0].cast<int>(), t[1].cast<T>(), t[2].cast<int>(), t[3].cast<int>());
    });
}

PYBIND11_MODULE(py_gpunufft_singlecoil_operator, m)
{
    declare_op<float>(m, "float");
}
