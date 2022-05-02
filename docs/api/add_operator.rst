Add new C++/CUDA operator
=========================

We show the sample usage of the C++/CUDA interface in writing a Scale Operator. 

Preparation
***************

First, you need to create a header file and a source file in ``src/operators/``.
In the next step, you have to add the newly created source and header file to `src/CMakeLists.txt` to the variables ``OPERATOR_HEADER`` and ``OPERATOR_SOURCE``.

.. code-block::

    .
    └── src             : OPTOX library source files
        └── operators   : actual implementation of operator functionality
            ├── scale_operator.h    : Header File of your new operator
            └── scale_operator.cu   : Source File of your new operator
        └── CMakeLists.txt  : Header and source files need to be added.

Implementation
***************

In the header file `src/operators/scale_operator.h` you have to first set the number or required inputs and outputs of your forward and adjoint path.
In the source file, you have to implement the ``computeForward`` and ``computeAdjoint`` method. Also, you should define your CUDA kernels there.
Let's have a look at the ``computeForward`` method for the scale operator. The method receives an ``OperatorOutputVector`` and ``OperatorInputVector``.
We assume that this operator has 1 input and 1 output. We get the respective input and
output via the ``getInput`` and ``getOutput`` methods that we call on the inputs / outputs vector. To fill a tensor with a constant value, 
simply call ``fill(<fill-value>)`` on the respective tensor. To define the ``dim_grid``, you can make
use of the ``optox`` function ``divUp``. Following example also shows how to perform :ref:`error checks<exceptions>` with ``OPTOX_CUDA_CHECK``, and how to throw an ``THROW_OPTOXEXCEPTION``.

.. code-block:: cpp

    template<typename T>
    void optox::ScaleOperator<T>::computeForward(optox::OperatorOutputVector &&outputs,
        const optox::OperatorInputVector &inputs)
    {
        // Extract input
        auto x = this->template getInput<T, 1>(0, inputs);
        // Extract output
        auto out = this->template getOutput<T, 1>(0, outputs);

        // Check sizes
        if (x->size() != out->size())
            THROW_OPTOXEXCEPTION("ScaleOperator: unsupported size");

        // Kernel specifications
        dim3 dim_block = dim3(128, 1, 1);
        dim3 dim_grid = dim3(divUp(out->size()[1], dim_block.x),
                            1,
                            1);

        // Clear the output
        out->fill(optox::type_trait<T>::make(0));

        // Launch CUDA kernel
        scale_kernel<T> <<<dim_grid, dim_block, 0, this->stream_>>>(*out, *x);
        OPTOX_CUDA_CHECK;
    }

Note also that this example is templated (``T``). This operator works for both complex-valued and real-valued types.
Although templates are mostly defined via ``float`` / ``double``,
we can estimate the respective complex-valued type using :ref:`typetraits` when using both real-valued and complex-valued tensors for the same call.
We specify the templates of the CUDA kernels for both types:

Real-valued CUDA kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

    template <typename T>
    __global__ void scale_kernel(
        typename optox::DTensor<typename optox::type_trait<T>::real_type, 1>::Ref out,
        const typename optox::DTensor<typename optox::type_trait<T>::real_type, 1>::ConstRef in)
    {
        // Get index
        int x = threadIdx.x + blockIdx.x * blockDim.x;

        // Use constant scale for now
        T scale = optox::type_trait<T>::make(2.0);

        // Check if indices are in range
        if (x < out.size_[0])
        {
            // Compute the corresponding index 
            out(x) = in(x) * scale;
        }
    }


Complex-valued CUDA kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

    template <typename T>
    __global__ void scale_kernel(
        typename optox::DTensor<typename optox::type_trait<T>::complex_type, 1>::Ref out,
        const typename optox::DTensor<typename optox::type_trait<T>::complex_type, 1>::ConstRef in)
    {
        // Get index
        int x = threadIdx.x + blockIdx.x * blockDim.x;

        // Use constant scale for now
        T scale = optox::type_trait<T>::make_complex(2.0, 3.0);

        // Check if indices are in range
        if (x < out.size_[0])
        {
            // Compute the corresponding index 
            out(x) = optox::complex_multiply<T>(in(x), scale);
        }
    }

These code blocks shows a sample CUDA kernel and the usage of the C++ :cpp:class:`optox::DTensor` class. We are able to access the 
:cpp:class:`optox::DTensor::Ref`'s size via its public member :cpp:member:`optox::DTensor::Ref::size_`. Furthermore, we can directly access the :cpp:class:`optox::DTensor`'s pixel values 
via brackets, similar to Matlab, as the brackets are overloaded. Hence, we do not require to pass any sizes, strides, etc. to 
the CUDA kernel, which keeps the argument list clean.

Finally, you have to register your class templates. You can make use of the built-in ``OPTOX_CALL_NUMBER_TYPES`` or ``OPTOX_CALL_REAL_NUMBER_TYPES`` defined in `src/operators/ioperator.h`.

.. code-block: cpp

    #define REGISTER_OP(T) \
        template class optox::ScaleOperator<T>;

    OPTOX_CALL_NUMBER_TYPES(REGISTER_OP);
    #undef REGISTER_OP