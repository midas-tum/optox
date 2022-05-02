Extend to Torch
====================

After writing the Scale Operator with the C++/CUDA interface, we now add the Torch wrapper for this operator. 

Preparation
***************

First, you need to create a C++ Python wrapper file `pytorch/th_scale_operator.cpp` and add the dependency to `pytorch/CMakeLists.txt` by using ``add_deps_th(scale)``.
Next, you need to create a python interface using `pytorch/optoth/scale.py` and add this module to `python/setup.py.in` using ``add_module("scale")``.
We also highly recommend to add unittesting. Therefore, you need to create a file `pytorch/test/test_scale.py`.

.. code-block::

    .
    └── pytorch
        └── optoth
            ├── scale.py    : Embed the *.so file, add the Python interface
            └── test        : Test folder
                └── test_scale.py   : Add unittesting (optional, but recommended)
        ├── th_scale_operator.cpp   : C++ Python Wrapper
        ├── CMakeLists.txt  : Add dependencies using add_deps_th(scale)
        └── setup.py.in     : Add the module using add_module("scale")


Implementation
***************

The Python wrapper is based on `pybind11`. Following steps have to be performed:

Declare the (templated) operator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Defines the class name how it appears in python, passes arguments to init if applicable, defined the foward/adjoint method.

.. code-block:: cpp

    template<typename T>
    void declare_op(py::module &m, const std::string &typestr)
    {
        std::string pyclass_name = std::string("Scale_") + typestr;
        py::class_<optox::ScaleOperator<T>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
        .def(py::init<>())
        .def("forward", forward<T>)
        .def("adjoint", adjoint<T>);
    }

Methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following codesnippet shows how the forward method is implemented for the Python wrapper.

.. code-block:: cpp

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


Create pybind11 module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When creating the pybind11 module. Also, you should register all your class templates here.

.. code-block:: cpp

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
    {
        declare_op<float>(m, "float");
        declare_op<double>(m, "double");
        declare_op<float2>(m, "float2");
        declare_op<double2>(m, "double2");
    }


Load extension
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the python file ``python/optopy/scale.py`` you can then load the extension and call the operators.

.. code-block:: python

    import _ext.th_scale_operator
    _ext.th_scale_operator.Scale_float().forward(x)

Also, you should define the ``torch.nn.Module`` and the ``torch.autograd.Function`` when you want to use this operator for training.

.. code-block:: python

    class ScaleFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.op = get_operator(x.dtype)
            shape = x.shape
            out = ctx.op.forward(x.flatten())
            return out.view(shape)

        @staticmethod
        def backward(ctx, grad_out):
            shape = grad_out.shape
            out = ctx.op.adjoint(grad_out.flatten())
            return out.view(shape)

    class Scale(torch.nn.Module):
        def forward(self, x):
            return ScaleFunction.apply(x)