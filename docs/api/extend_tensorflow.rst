Extend to Tensorflow / Keras
================================

After writing the Scale Operator with the C++/CUDA interface, we now add the Tensorflow wrapper for this operator. 

Preparation
***************

First, you need to create a C++ Python wrapper file `tensorflow/tf_scale_operator.cpp` and add the dependency to `tensorflow/CMakeLists.txt` by using ``finish_tensorflow_lib(scale)``.
Next, you need to create a python interface using `tensorflow/optotf/scale/__init__.py` and `tensorflow/optotf/keras/scale.py` and add this module in `tensorflow/setup.py.in` to the ``ADD_PACKAGES`` list.
We also highly recommend to add unittesting. Therefore, you need to create a file `tensorflow/test/test_scale.py` and `tensorflow/test/test_scale_keras.py`.

.. code-block::

    .
    └── tensorflow
        └── optotf
            ├── scale       : Embed the *.so file, add the Python interface
            |   └── __init__.py     : Register gradients, tensorflow interface
            ├── keras
            |   └── scale.py        : Keras interface
            └── test        : Test folder
                └── test_scale.py   : Add unittesting (optional, but recommended)
        ├── tf_scale_operator.cpp   : C++ Python Wrapper
        ├── CMakeLists.txt  : Add dependencies using finish_tensorflow_lib(scale)
        └── setup.py.in     : Add the module to the ADD_PACKAGES list.

Implementation
***************

The Tensorflow wrapper builds on the Tensorflow framework and thus behaves differently than the Python and Torch wrappers that are using pybind11.
Following steps have to be performed:

Declare the (templated) operator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cpp

    REGISTER_OP("Scale")
        .Attr("T: numbertype")
        .Input("x: T")
        .Output("output: T")
        .SetShapeFn(shape_inference::UnchangedShape);

The ``.SetShapeFn`` is crucial such that the static graph can be built correctly!

Define operator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cpp

    template <typename T>
    class TFScaleOperator : public OpKernel {
    public:
        
        explicit TFScaleOperator(OpKernelConstruction* context) 
            : OpKernel(context)
        {}

        void Compute(OpKernelContext* context) override {
            // Grab the input tensor
            const Tensor& x_tensor = context->input(0);
            TensorShape output_shape = x_tensor.shape();

            // allocate the output
            Tensor* output_tensor = nullptr;
            OP_REQUIRES_OK(context,
                context->allocate_output(0, output_shape, &output_tensor));

            // compute the output
            auto input = getDTensorTensorflow<T, 1>(x_tensor);
            auto output = getDTensorTensorflow<T, 1>(*output_tensor);
            
            optox::ScaleOperator<T> op;
            op.setStream(context->eigen_device<GPUDevice>().stream());
            op.forward({output.get()}, {input.get()});
        }
    };

Register operator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cpp

    #define REGISTER_GPU(dtype) \
        REGISTER_KERNEL_BUILDER( \
            Name("Scale") \
            .Device(DEVICE_GPU) \
            .TypeConstraint<typename optox::tf<dtype>::type >("T"), \
            TFScaleOperator<dtype>) \

    REGISTER_GPU(float);
    REGISTER_GPU(double);
    REGISTER_GPU(float2);
    REGISTER_GPU(double2);

    #undef REGISTER_GPU

Load extension and register gradients
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the python file ``tensorflow/optotf/scale/__init__.py`` you can then load the extension and register the gradients such that the operator can be used for training.

.. code-block:: python

    _ext = tf.load_op_library(tf.compat.v1.resource_loader.get_path_to_datafile("tf_scale_operator.so"))

    @_ops.RegisterGradient("Scale")
    def _scale_forward_grad(op, grad):
        grad_in = _ext.scale_grad(grad)
        return [grad_in]

Define Keras interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the python file ``tensorflow/optotf/keras/scale.py`` you can finally define the Keras interface.

.. code-block:: python
    
    import tensorflow as tf
    import optotf.scale

    class Scale(tf.keras.layers.Layer):
        def __init__(self):
            super().__init__()
            self.op = lambda x: optotf.scale.scale(x)

        def call(self, x):
            self.op(x)