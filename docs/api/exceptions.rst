.. _exceptions:

Exceptions and Error Checking
=============================

Throw an exception related to ``optox``:

.. code-block:: cpp

    THROW_OPTOXEXCEPTION("My exception message");

Check if CUDA function finished safely:

.. code-block:: cpp

    OPTOX_CUDA_CHECK;

Safely call a CUDA function:

.. code-block:: cpp

    OPTOX_CUDA_SAFE_CALL(<cuda_function>);

Safely call a CUFFT function:

.. code-block:: cpp

    OPTOX_CUFFT_SAFE_CALL(<cufft_function>);