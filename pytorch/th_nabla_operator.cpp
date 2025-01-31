///@file th_nabla_operator.cpp
///@brief PyTorch wrappers for nabla operator
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 02.2019

#include <vector>

#include "th_utils.h"
#include "operators/nabla_operator.h"
#include "operators/nabla2_operator.h"

#include <torch/extension.h>
#include <pybind11/pybind11.h>


template<typename T, int N>
at::Tensor forward(optox::NablaOperator<T, N> &op, at::Tensor th_input)
{
    // parse the input tensors
    auto input = getDTensorTorch<T, N>(th_input);

    // allocate the output tensor
    std::vector<int64_t> shape;
    shape.push_back(N);
    auto in_shape = th_input.sizes().vec();
    shape.insert(shape.end(), in_shape.begin(), in_shape.end());
    auto th_output = at::empty(shape, th_input.options());
    auto output = getDTensorTorch<T, N+1>(th_output);

    op.forward({output.get()}, {input.get()});
    return th_output;
}

template<typename T, int N>
at::Tensor adjoint(optox::NablaOperator<T, N> &op, at::Tensor th_input)
{
    // parse the input tensors
    auto input = getDTensorTorch<T, N+1>(th_input);

    // allocate the output tensor
    std::vector<int64_t> shape;
    auto in_shape = th_input.sizes().vec();
    shape.insert(shape.end(), in_shape.begin()+1, in_shape.end());
    auto th_output = at::empty(shape, th_input.options());
    auto output = getDTensorTorch<T, N>(th_output);
    
    op.adjoint({output.get()}, {input.get()});

    return th_output;
}

template<typename T, int N>
at::Tensor forward2(optox::Nabla2Operator<T, N> &op, at::Tensor th_input)
{
    // parse the input tensors
    auto input = getDTensorTorch<T, N+1>(th_input);

    // allocate the output tensor
    std::vector<int64_t> shape;
    shape.push_back(N*N);
    auto in_shape = th_input.sizes().vec();
    shape.insert(shape.end(), in_shape.begin()+1, in_shape.end());
    auto th_output = at::empty(shape, th_input.options());
    auto output = getDTensorTorch<T, N+1>(th_output);

    op.forward({output.get()}, {input.get()});

    return th_output;
}

template<typename T, int N>
at::Tensor adjoint2(optox::Nabla2Operator<T, N> &op, at::Tensor th_input)
{
    // parse the input tensors
    auto input = getDTensorTorch<T, N+1>(th_input);

    // allocate the output tensor
    std::vector<int64_t> shape;
    shape.push_back(N);
    auto in_shape = th_input.sizes().vec();
    shape.insert(shape.end(), in_shape.begin()+1, in_shape.end());
    auto th_output = at::empty(shape, th_input.options());
    auto output = getDTensorTorch<T, N+1>(th_output);
    
    op.adjoint({output.get()}, {input.get()});

    return th_output;
}

template<typename T, int N>
void declare_op(py::module &m, const std::string &typestr)
{
    std::string pyclass_name = std::string("Nabla_") + std::to_string(N) + "d_" + typestr;
    py::class_<optox::NablaOperator<T, N>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
    .def(py::init<const T&, const T&, const T&, const T&>(), py::arg("hx") = 1.0, py::arg("hy") = 1.0, py::arg("hz") = 1.0, py::arg("ht") = 1.0)
    .def("forward", forward<T, N>)
    .def("adjoint", adjoint<T, N>);

    pyclass_name = std::string("Nabla2_") + std::to_string(N) + "d_" + typestr;
    py::class_<optox::Nabla2Operator<T, N>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
    .def(py::init<>())
    .def("forward", forward2<T, N>)
    .def("adjoint", adjoint2<T, N>);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    declare_op<float, 2>(m, "float");
    declare_op<double, 2>(m, "double");

    declare_op<float, 3>(m, "float");
    declare_op<double, 3>(m, "double");

    declare_op<float, 4>(m, "float");
    declare_op<double, 4>(m, "double");
}
