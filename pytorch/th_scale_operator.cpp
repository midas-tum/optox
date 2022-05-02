///@file th_scale_operator.cpp
///@brief torch wrappers for the scale operator
///@author Kerstin Hammernik <k.hammernik@imperial.ac.uk>
///@date 03.2022

#include "th_utils.h"
#include "operators/scale_operator.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

template<typename T>
at::Tensor forward(optox::ScaleOperator<T> &op, at::Tensor th_input)
{
    // parse the input tensors
    auto input = getDTensorTorch<T, 1>(th_input);

    // allocate the output tensor
    std::vector<int64_t> shape;
    shape.push_back(input->size()[0]);

    auto th_output = at::empty(shape, th_input.options());
    auto output = getDTensorTorch<T, 1>(th_output);

    op.forward({output.get()}, {input.get()});
 
    return th_output;
}

template<typename T>
at::Tensor adjoint(optox::ScaleOperator<T> &op, at::Tensor th_input)
{
    // parse the input tensors
    auto input = getDTensorTorch<T, 1>(th_input);

    // allocate the output tensor
    std::vector<int64_t> shape;
    shape.push_back(input->size()[0]);

    auto th_output = at::empty(shape, th_input.options());
    auto output = getDTensorTorch<T, 1>(th_output);

    op.adjoint({output.get()}, {input.get()});

    return th_output;
}

template<typename T>
void declare_op(py::module &m, const std::string &typestr)
{
    std::string pyclass_name = std::string("Scale_") + typestr;
    py::class_<optox::ScaleOperator<T>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
    .def(py::init<>())
    .def("forward", forward<T>)
    .def("adjoint", adjoint<T>);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    declare_op<float>(m, "float");
    declare_op<double>(m, "double");
    declare_op<float2>(m, "float2");
    declare_op<double2>(m, "double2");
}
