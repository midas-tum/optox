# Operator to X - `optox`
## Goal
Write operators only once and use it **everywhere**. 

## Concept
Write an operator in `C++/CUDA` and generate wrappers to different languages such as `Python` and machine learning libraries such as `Tensorflow` or `Pytorch`.

`optox` provides a tensor interface to ease data transfer between host tensors `optox::HTensor` and device tensors `optox::DTensor` of any floating type and number of dimensions.
Using this interface, an operator is only written once in `C++/CUDA` and wrappers for `Python`, `Tensorflow 2.x` and `Pytorch` expose the functionality to a higher level application (e.g. iterative reconstruction, custom deep learning reconstruction, ...).

## Overview 
The source files are organized as follows:

    .
    +-- src             : `optox` library source files
    |   +-- tensor      : header only implementation of `HTensor` and `DTensor`
    |   +-- operators   : actual implementation of operator functionality
    +-- python          : python wrappers 
    +-- pytorch         : pytorch wrappers
    +-- tensorflow      : tensorflow wrappers

## Install instructions
### Automatic installation
We provide a python installation script. To build the respective packages for python, pytorch and tensorflow, simply add the flags `--python`, `--pytorch`, `--tensorflow`.
To build optox with [gpuNUFFT](https://github.com/khammernik/gpuNUFFT/tree/cuda_streams), you only need to add the flag `--gpunufft`.
```
git clone https://github.com/midas-tum/optox.git
cd optox
# build all
python install.py --python --pytorch --tensorflow --gpunufft
# build only with pytorch support
python install.py --pytorch
```

### Manual installation
**We highly recommend to use Cuda >11.1 which runs smoothly with both Pytorch and Tensorflow**

First setup the following environment variables:
- `CUDA_BIN_PATH` to point to the NVidia CUDA toolkit (typically `/usr/local/cuda`)

Note that the CUDA version used to build the `optox` library should match the version required by `Tensorflow` and/or `Pytorch`.

We provide an anaconda environment with `Tensorflow 2.4`, `Pytorch 1.9`, `Cuda 11.1`. The environment `optox` can be created via
```
conda env create -f environment.yml
```

To build the basic optox library perform the following steps:
```bash
mkdir build
cd build
cmake .. 
make install
```

### Troubleshooting
If the default gcc compiler version is >8, you will get a build error. A simple workaround is to call following before the cmake command and with a clean build dir, assuming `gcc-8` and `g++-8` are installed on your system in `/usr/bin/`
```
export CC=/usr/bin/gcc-8
export CXX=/usr/bin/g++-8
```

If you experience a problem that lcudart library cannot be found, then link build to the location of the CUDA lib64 libraries (typically `<path-to-cuda-lib64> = /usr/local/cuda/lib64`)
```
export LDFLAGS=-L<path-to-cuda-lib64>
```

### CUDA sync free build
Use `CMAKE_BUILD_TYPE=Release` to avoid the device synchronization after each CUDA call. Then, no CUDA errors are generated but runtime is strongly reduced.

### `Python` wrappers
To build the `Python` wrappers `optox` requires `pybind11` which can be installed in an anaconda environment by `conda install pybind11`.
To also build `Python` wrappers substitute the `cmake` command by:
```bash
cmake .. -DWITH_PYTHON=ON
```

### `Pytorch` wrappers
To build it, the `pytorch` package must be installed.
```bash
cmake .. -DWITH_PYTORCH=ON
```
### `Tensorflow` wrappers
To build it, the `tensorflow` package must be installed.
```bash
cmake .. -DWITH_TENSORFLOW=ON
```

Note that multiple combinations are supported.

### `gpuNUFFT` wrappers
`optox` provides python, pytorch and tensorflow/keras wrappers for [khammernik/gpuNUFFT branch cuda_streams](https://github.com/khammernik/gpuNUFFT/tree/cuda_streams). Make sure to build gpuNUFFT in Release mode.
```
git clone --branch cuda_streams https://github.com/khammernik/gpuNUFFT.git
cd gpuNUFFT/CUDA
mkdir -p build
cd build
cmake .. -DGEN_MEX_FILES=OFF -DCMAKE_BUILD_TYPE=Release
make
```
To build them with optox, the `gpuNUFFT` package must be built and the path `GPUNUFFT_ROOT_DIR` (e.g. `GPUNUFFT_ROOT_DIR=~/gpuNUFFT`)
must be set.
```
cmake .. -DWITH_GPUNUFFT=ON
```


## Testing

### `Python`
To perform an adjointness test of the `nabla` operator using the `Python` wrappers execute
```bash
python -m unittest optopy.nabla

```
If successful the output should be 
```bash
(optox) ∂ python -m unittest optopy.nabla 
dtype: <class 'numpy.float64'> dim: 2 diff: 6.661338147750939e-16
.dtype: <class 'numpy.float64'> dim: 3 diff: 2.842170943040401e-14
.dtype: <class 'numpy.float32'> dim: 2 diff: 2.86102294921875e-06
.dtype: <class 'numpy.float32'> dim: 3 diff: 7.62939453125e-06
.
----------------------------------------------------------------------
Ran 4 tests in 1.099s

OK

```


### `Pytorch`
To perform a gradient test of the `activations` operators using the `Pytorch` wrappers execute
```bash
python -m unittest optoth.activations.act

```
If successful the output should be 
```bash
(optox) ∂ python -m unittest optoth.activations.act 
grad_x: -3616.3090656 num_grad_x -3616.3090955 success: True
grad_w: 7232.6181312 num_grad_w 7232.6181312 success: True
.grad_x: 535.2185935 num_grad_x 535.2185935 success: True
grad_w: 2236.8791233 num_grad_w 2236.8791233 success: True
.grad_x: -215.0009414 num_grad_x -215.0009432 success: True
grad_w: 430.0018828 num_grad_w 430.0018828 success: True
.
----------------------------------------------------------------------
Ran 3 tests in 2.263s

OK

```


### `Tensorflow`
To perform an adjointness test of the `nabla` operators using the `Tensorflow` wrappers execute
```bash
python -m unittest optotf.nabla

```
If successful the output should be 
```bash
(optox) ∂ python -m unittest optotf.nabla
...
dtype: <dtype: 'float64'> dim: 2 diff: 1.0658141036401503e-14
.dtype: <dtype: 'float32'> dim: 2 diff: 0.0
.
----------------------------------------------------------------------
Ran 2 tests in 1.490s


OK

```

### `Keras` support
The keras layers can be found in `optotf.keras.xxx`.

## Unittests
Unittests can be called as follows:
```
python -m unittest discover optotf.test
python -m unittest discover optoth.test
python -m unittest discover optopy.test
```