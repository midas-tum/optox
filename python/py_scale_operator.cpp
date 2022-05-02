///@file py_scale_operator.cpp
///@brief python wrappers for the scale operator
///@author Kerstin Hammernik <k.hammernik@imperial.ac.uk>
///@date 03.2022

#include <vector>

#include "py_utils.h"
#include "operators/scale_operator.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

template<typename T>
py::array forward(optox::ScaleOperator<T> &op, py::array np_input)
{
    // Parse the input tensors
    auto input = getDTensorNp<T, 1>(np_input);

    // Init the output tensors
    optox::DTensor<T, 1> output(input->size());

    // Compute forward
    op.forward({&output}, {input.get()});

    // Wrap the output tensor into a numpy array
    return dTensorToNp<T, 1>(output); 
}

template<typename T>
py::array adjoint(optox::ScaleOperator<T> &op, py::array np_input)
{
    // Parse the input tensors
    auto input = getDTensorNp<T, 1>(np_input); 

    optox::DTensor<T, 1> output(input->size());

    op.forward({&output}, {input.get()});

    return dTensorToNp<T, 1>(output);
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

PYBIND11_MODULE(py_scale_operator, m)
{
    declare_op<float>(m, "float");
    declare_op<double>(m, "double");
    declare_op<float2>(m, "float2");
    declare_op<double2>(m, "double2");
}
