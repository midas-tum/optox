///@file py_warp_operator.cpp
///@brief python wrappers for the warp operator
///@author Kerstin Hammernik <k.hammernik@tum.de>
///@date 12.2020

#include <vector>

#include "py_utils.h"
#include "operators/warp_operator.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

template<typename T>
std::vector<py::array> forward(optox::WarpOperator<T> &op, py::array np_grad_out, py::array np_u, py::array np_x)
{
    // parse the input tensors
    auto x = getDTensorNp<T, 4>(np_x);
    auto u = getDTensorNp<T, 4>(np_u);
    auto grad_out = getDTensorNp<T, 4>(np_grad_out);

    // allocate output tensor
    optox::DTensor<T, 4> output(x->size());
    optox::DTensor<T, 4> grad_u(u->size());

    op.forward({&output, &grad_u}, {grad_out.get(), u.get(), x.get()});
    return {dTensorToNp<T, 4>(output), dTensorToNp<T, 4>(grad_u)};
}

template<typename T>
std::vector<py::array> adjoint(optox::WarpOperator<T> &op, py::array np_grad_out, py::array np_u, py::array np_x)
{
    // parse the input tensors
    auto grad_out = getDTensorNp<T, 4>(np_grad_out);
    auto u = getDTensorNp<T, 4>(np_u);
    auto x = getDTensorNp<T, 4>(np_x);

    // allocate the output tensor
    optox::DTensor<T, 4> grad_x(grad_out->size());
    optox::DTensor<T, 4> grad_u(u->size());

    op.adjoint({&grad_x, &grad_u}, {grad_out.get(), u.get(), x.get()});

    return {dTensorToNp<T, 4>(grad_x), dTensorToNp<T, 4>(grad_u)};
}

template<typename T>
void declare_op(py::module &m, const std::string &typestr)
{
    std::string pyclass_name = std::string("Warp_2d_") + typestr;
    py::class_<optox::WarpOperator<T>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
    .def(py::init<const std::string&>())
    .def("forward", forward<T>) 
    .def("adjoint", adjoint<T>);
}

PYBIND11_MODULE(py_warp_operator, m)
{
    declare_op<float>(m, "float");
    declare_op<double>(m, "double");
}
