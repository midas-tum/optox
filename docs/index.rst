.. OPTOX documentation master file, created by
   sphinx-quickstart on Thu Feb 24 15:26:23 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to OPTOX's documentation!
=================================

Write an operator in C++/CUDA and generate wrappers to different languages such as Python and machine learning libraries such as Tensorflow or Pytorch.

OPTOX provides a tensor interface to ease data transfer between host tensors :cpp:class:`optox::HTensor` and device tensors :cpp:class:`optox::DTensor` of any floating type and number of dimensions.
Using this interface, an operator is only written once in C++/CUDA and wrappers for Python, Tensorflow 2.x and Pytorch expose the functionality to a higher level application (e.g. iterative reconstruction, custom deep learning reconstruction, ...).

OVERVIEW
--------
The source files are organized as follows:

.. code-block::

    .
    ├── src             : OPTOX library source files
    |   ├── tensor      : header only implementation of optox::HTensor and optox::DTensor
    |   └── operators   : actual implementation of operator functionality
    ├── python          : python wrappers 
    ├── pytorch         : pytorch wrappers
    └── tensorflow      : tensorflow wrappers

.. toctree::
   :caption: USE OPTOX
   :titlesonly:
   :maxdepth: 1

   installation
   getting_started
   unittesting

.. toctree::
    :maxdepth: 2
    :caption: Python API

    api/optopy
    api/optoth
    api/optotf

.. toctree::
   :maxdepth: 2
   :caption: C++ API

   api/tensor
   api/shape
   api/operator
   api/typetraits
   api/exceptions

.. toctree::
   :caption: Extend optox
   :titlesonly:
   :maxdepth: 1

   api/add_operator
   api/extend_python
   api/extend_tensorflow
   api/extend_pytorch

.. toctree::
   :caption: About
   :titlesonly:
   :maxdepth: 1

   about

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

