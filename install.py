import os
import shutil
import pathlib
import sys
import subprocess
from subprocess import Popen, PIPE
import argparse
import multiprocessing

argparser = argparse.ArgumentParser(add_help=False)
argparser.add_argument('--gpunufft', help='gpuNUFFT', action="store_true")
argparser.add_argument('--pytorch', help='pytorch', action="store_true")
argparser.add_argument('--python', help='python', action="store_true")
argparser.add_argument('--tensorflow', help='Tensorflow', action="store_true")

args, unknown = argparser.parse_known_args()
sys.argv = [sys.argv[0]] + unknown

def with_arg(boolean_value):
    return "ON" if boolean_value else "OFF"
with_tensorflow = with_arg(args.tensorflow)
with_pytorch = with_arg(args.pytorch)
with_python = with_arg(args.python)

# look for CUDA environment
cuda_bin_path = os.environ.get('CUDA_BIN_PATH')
if not cuda_bin_path:
    inval = input('CUDA_BIN_PATH not specified [/usr/local/cuda]: ')
    if inval:
        cuda_bin_path = inval
    else:
        cuda_bin_path = '/usr/local/cuda'
    os.environ['CUDA_BIN_PATH'] = cuda_bin_path
os.environ['CUDA_ROOT_DIR'] = cuda_bin_path
os.environ['CUDA_HOME'] = cuda_bin_path
os.environ['CUDA_LIB_PATH'] = cuda_bin_path+'/lib64'
if os.environ.get('LD_LIBRARY_PATH'):
    os.environ['LD_LIBRARY_PATH'] += os.environ['CUDA_LIB_PATH'] + ':'
else:
    os.environ['LD_LIBRARY_PATH'] = os.environ['CUDA_LIB_PATH'] + ':'
os.environ['CPLUS_INCLUDE_PATH'] = cuda_bin_path+'/include'
os.environ['LDFLAGS'] = '-L'+cuda_bin_path+'/lib64'
print('CUDA_BIN_PATH: ' + os.environ['CUDA_BIN_PATH'])
print('LD_LIBRARY_PATH: ' + os.environ['LD_LIBRARY_PATH'])
print('LDFLAGS: ' + os.environ['LDFLAGS'])

# check GCC/G++ version
p = Popen(['gcc', '--version'], stdout=PIPE, stderr=PIPE, stdin=PIPE)
gcc_version = [s for s in p.stdout.read().decode('utf-8').split('\n') if 'gcc' in s][0].split(' ')[-1]
if gcc_version.split('.')[0] > "8":
    # GCC v8 is required for compilation, check if provided
    p = Popen(['which', 'gcc-8'], stdout=PIPE, stderr=PIPE, stdin=PIPE)
    gcc8 = p.stdout.read().decode('utf-8').split('\n')[0]
    if not gcc8:
        sys.exit('GCC version {} > 8.X.X: GCC-8 is required for compilation, please either install in conda environment (conda env create -f environment.yml) or system-wide'.format(gcc_version))
    else:
        if not os.environ.get('CC'):
            os.environ['CC'] = gcc8
p = Popen(['g++', '--version'], stdout=PIPE, stderr=PIPE, stdin=PIPE)
gcc_version = [s for s in p.stdout.read().decode('utf-8').split('\n') if 'g++' in s][0].split(' ')[-1]
if gcc_version.split('.')[0] > "8":
    p = Popen(['which', 'g++-8'], stdout=PIPE, stderr=PIPE, stdin=PIPE)
    gxx8 = p.stdout.read().decode('utf-8').split('\n')[0]
    if not gxx8:
        sys.exit('G++ version {} > 8.X.X: G++-8 is required for compilation, please either install in conda environment (conda env create -f environment.yml) or system-wide'.format(gcc_version))
    else:
        if not os.environ.get('CXX'):
            os.environ['CXX'] = gxx8

with open(str(pathlib.Path().absolute()) + '/requirements.txt') as f:
    lines = f.readlines()

lines = [line.split('\n')[0] for line in lines]
if with_tensorflow == 'OFF':
    lines = [line for line in lines if 'tensorflow' not in line]

if with_pytorch == 'OFF':
    lines = [line for line in lines if 'torch' not in line]

cwd = pathlib.Path().absolute()
build_temp = pathlib.Path('./build')
build_temp.mkdir(parents=True, exist_ok=True)

with open(str(pathlib.Path().absolute()) + '/build/requirements.txt', 'w') as f:
    for line in lines:
        f.writelines(line + '\n')
subprocess.run(['pip install -r ' + str(pathlib.Path().absolute()) + '/build/requirements.txt'], shell=True)
cmake_args = [
       f'-DWITH_TENSORFLOW={with_tensorflow}',
       f'-DWITH_PYTORCH={with_pytorch}',
       f'-DWITH_PYTHON={with_python}']

if args.gpunufft:
    print("use gpunufft...")
    subprocess.run(["git", "clone", "--branch", "cuda_streams", "https://github.com/khammernik/gpuNUFFT.git"])
    os.chdir(str(cwd) + "/gpuNUFFT/CUDA")
    subprocess.run(["mkdir", "-p", "build"])
    os.chdir(str(cwd) + "/gpuNUFFT/CUDA/build")
    subprocess.run(["cmake", str(cwd) + "/gpuNUFFT/CUDA", "-DGEN_MEX_FILES=OFF"])
    subprocess.run(["make"])
    os.chdir(str(cwd))
    cmake_args += ['-DWITH_GPUNUFFT=ON']
    if not os.environ.get('GPUNUFFT_ROOT_DIR'):
        os.environ['GPUNUFFT_ROOT_DIR'] = os.path.join(str(cwd), 'gpuNUFFT')
    print('GPUNUFFT_ROOT_DIR: ' + os.environ['GPUNUFFT_ROOT_DIR'])
else:
    print("no gpunufft...")

build_args = []

os.chdir(str(build_temp))
ncpus = multiprocessing.cpu_count()
subprocess.run(['cmake', str(cwd)] + cmake_args)
subprocess.run(["make", f"-j{ncpus}"])
subprocess.run(["make", "install"])
os.chdir(str(cwd))
shutil.rmtree(str(build_temp))
