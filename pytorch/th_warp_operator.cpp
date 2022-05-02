///@file th_warp_operator.cpp
///@brief PyTorch wrappers for warp operator
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 09.2019

#include <vector>

#include "th_utils.h"
#include "operators/warp_operator.h"

#include <torch/extension.h>
#include <pybind11/pybind11.h>


template<typename T>
std::vector<at::Tensor> forward(optox::WarpOperator<T> &op, at::Tensor th_grad_out, at::Tensor th_u, at::Tensor th_x)
{
    // parse the tensors
    auto x = getDTensorTorch<T, 4>(th_x);
    auto u = getDTensorTorch<T, 4>(th_u);
    auto grad_out = getDTensorTorch<T, 4>(th_grad_out);

    // allocate the output tensor
    at::Tensor th_out = at::empty_like(th_x);
    auto out = getDTensorTorch<T, 4>(th_out);
    at::Tensor th_grad_u = at::empty_like(th_u);
    auto grad_u = getDTensorTorch<T, 4>(th_grad_u);

    op.forward({out.get(), grad_u.get()}, {grad_out.get(), u.get(), x.get()});

    return {th_out, th_grad_u};
}

template<typename T>
std::vector<at::Tensor> adjoint(optox::WarpOperator<T> &op, at::Tensor th_grad_out, at::Tensor th_u, at::Tensor th_x)
{
    // parse the tensors
    auto grad_out = getDTensorTorch<T, 4>(th_grad_out);
    auto u = getDTensorTorch<T, 4>(th_u);
    auto x = getDTensorTorch<T, 4>(th_x);

    // allocate the output tensor
    at::Tensor th_grad_x = at::empty_like(th_grad_out);
    auto grad_x = getDTensorTorch<T, 4>(th_grad_x);
    at::Tensor th_grad_u = at::empty_like(th_u);
    auto grad_u = getDTensorTorch<T, 4>(th_grad_u);

    op.adjoint({grad_x.get(), grad_u.get()}, {grad_out.get(), u.get(), x.get()});

    return {th_grad_x, th_grad_u};
}

template<typename T>
void declare_op(py::module &m, const std::string &typestr)
{
    std::string pyclass_name = std::string("Warp_") + typestr;
    py::class_<optox::WarpOperator<T>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
    .def(py::init<const std::string&>())
    .def("forward", forward<T>)
    .def("adjoint", adjoint<T>);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    declare_op<float>(m, "float");
    declare_op<double>(m, "double");
}
