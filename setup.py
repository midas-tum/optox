import os
import pathlib
import sys
import subprocess
import argparse


argparser = argparse.ArgumentParser(add_help=False)
args, unknown = argparser.parse_known_args()
sys.argv = [sys.argv[0]] + unknown
subprocess.run(['pip install -r ' + str(pathlib.Path().absolute()) + '/requirements.txt'], shell=True)


cwd = pathlib.Path().absolute()
build_temp = pathlib.Path('./build')
build_temp.mkdir(parents=True, exist_ok=True)

cmake_args = [
        '-DWITH_TENSORFLOW=ON -DWITH_PYTORCH=ON -DWITH_PYTHON=ON'
    ]

build_args = []

os.chdir(str(build_temp))
subprocess.run(['cmake', str(cwd)] + cmake_args)
subprocess.run(['cmake', '--build', '.'] + build_args)
subprocess.run(["make", "install"])
os.chdir(str(cwd))