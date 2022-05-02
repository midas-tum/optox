Extend to Python
====================

After writing the Scale Operator with the C++/CUDA interface, we now add the Python wrapper for this operator. 

Preparation
***************

First, you need to create a C++ Python wrapper file `python/py_scale_operator.cpp` and add the dependency to `python/CMakeLists.txt` by using ``add_deps_py(scale)``.
Next, you need to create a python interface using `python/optopy/scale.py` and add this module to `python/setup.py.in` using ``add_module("scale")``.
We also highly recommend to add unittesting. Therefore, you need to create a file `python/test/test_scale.py`.

.. code-block::

    .
    └── python
        └── optopy
            ├── scale.py    : Embed the *.so file, add the Python interface
            └── test        : Test folder
                └── test_scale.py   : Add unittesting (optional, but recommended)
        ├── py_scale_operator.cpp   : C++ Python Wrapper
        ├── CMakeLists.txt  : Add dependencies using add_deps_py(scale)
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

Create pybind11 module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When creating the pybind11 module, please check the correct name of the module, i.e., `*.so` file as indicated below.
Also, you should register all your class templates here.

.. code-block:: cpp

    PYBIND11_MODULE(py_scale_operator, m)
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

    import _ext.py_scale_operator
    _ext.py_scale_operator.Scale_float().forward(x)
