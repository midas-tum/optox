Installation
============

Optox environment
******************

- Make conda environment: ``conda env create -f environment.yml``
- Go to ``build``
- Configure without `gpuNUFFT <https://github.com/khammernik/gpuNUFFT>`_:
    - Run cmake: ``cmake ..``
- Configure with `gpuNUFFT <https://github.com/khammernik/gpuNUFFT>`_
    - Checkout the branch ``cuda_streams`` of the `gpuNUFFT <https://github.com/khammernik/gpuNUFFT>`_ repo.
    - Follow the guidelines to build `gpuNUFFT <https://github.com/khammernik/gpuNUFFT>`_ and make sure to build in `Release` mode
    - To configure ``optox`` with `gpuNUFFT <https://github.com/khammernik/gpuNUFFT>`_, run ``cmake .. -DWITH_GPUNUFFT=ON``
- Compile: ``make -j``
- Install: ``make install``

Documentation
***************

- Make sure to have Doxygen installed: ``sudo apt install doxygen``
- Make sure to have the requirements installed: ``pip install -r requirements.txt``
- Go to ``docs``
- Run ``make html``
- Go to ``_build/html``
- View the documentation: ``python -m http.server 8000``