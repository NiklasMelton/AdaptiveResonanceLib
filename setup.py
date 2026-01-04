# setup.py
import sys
import os
import pybind11
from setuptools import setup, Extension, find_packages

extra_compile_args = ["/std:c++17"] if sys.platform == "win32" else ["-std=c++17"]

cpp_dir = os.path.join("artlib", "optimized", "backends", "cpp")

ext_modules = [
    Extension(
        "artlib.optimized.backends.cpp.cppBinaryFuzzyARTMAP",
        [os.path.join(cpp_dir, "cppBinaryFuzzyARTMAP.cpp")],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        "artlib.optimized.backends.cpp.cppFuzzyARTMAP",
        [os.path.join(cpp_dir, "cppFuzzyARTMAP.cpp")],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        "artlib.optimized.backends.cpp.cppHypersphereARTMAP",
        [os.path.join(cpp_dir, "cppHypersphereARTMAP.cpp")],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        "artlib.optimized.backends.cpp.cppGaussianARTMAP",
        [os.path.join(cpp_dir, "cppGaussianARTMAP.cpp")],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        "artlib.optimized.backends.cpp.cppART1MAP",
        [os.path.join(cpp_dir, "cppART1MAP.cpp")],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        "artlib.optimized.backends.cpp.fracsort",
        [os.path.join(cpp_dir, "fracsort.cpp")],
        include_dirs=[pybind11.get_include(), cpp_dir],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        "artlib.optimized.backends.cpp.cppFuzzyART",
        [os.path.join(cpp_dir, "cppFuzzyART.cpp")],
        include_dirs=[pybind11.get_include(), cpp_dir],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        "artlib.optimized.backends.cpp.cppBinaryFuzzyART",
        [os.path.join(cpp_dir, "cppBinaryFuzzyART.cpp")],
        include_dirs=[pybind11.get_include(), cpp_dir],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        "artlib.optimized.backends.cpp.cppART1",
        [os.path.join(cpp_dir, "cppART1.cpp")],
        include_dirs=[pybind11.get_include(), cpp_dir],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
]

setup(
    name="artlib",
    version="0.1.7",
    packages=find_packages(),  # This all are included
    ext_modules=ext_modules,
)
