.. _typetraits:

Typetraits
=============================

The header file `src/typetraits.h` contains wrappers to allow for more flexible templating when vector-valued types are involved, i.e., complex numbers.
Complex numbers are represented as ``float2`` and ``double2``, respectively. We often template for ``float`` and ``double`` and want to get the corresponding complex type.
We can typedef as follows:

.. code-block:: cpp

    typedef typename optox::type_trait<float>::complex_type complex_type; // results in float2
    typedef typename optox::type_trait<double>::complex_type complex_type; // results in double2

In case a template parameter ``T`` is present, the line changes to:

.. code-block:: cpp

    // results in float2 for T=float and double2 for T=double
    typedef typename optox::type_trait<T>::complex_type complex_type; 

We can also created templated complex numbers as follows:

.. code-block:: cpp

    // init real and imaginary part with zero
    complex_type fill_value = optox::type_trait<T>::make_complex(0);
