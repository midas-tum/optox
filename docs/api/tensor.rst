.. _tensor:

Tensor Class
=============

Tensors are the core of the OPTOX framework and handle all memory management.
It is also possible to only wrap a data pointer.
Tensors are defined for host and device in :cpp:class:`optox::HTensor` and :cpp:class:`optox::DTensor`, respectively.


Tensor
-------
Defined in ``tensor/tensor.h``

.. doxygenclass:: optox::ITensor
   :project: optox

.. doxygenclass:: optox::Tensor
   :project: optox

HTensor
--------

Defined in ``tensor/h_tensor.h``

.. doxygenclass:: optox::HTensor
   :project: optox

DTensor
--------

Defined in ``tensor/d_tensor.h``

.. doxygenclass:: optox::DTensor
   :project: optox

