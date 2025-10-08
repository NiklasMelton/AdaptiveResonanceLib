# setup.py
import sys
import os
import pybind11
from setuptools import setup, Extension, find_packages

extra_compile_args = ["/std:c++17"] if sys.platform == "win32" else ["-std=c++17"]

ext_modules = [
    Extension(
        "artlib.optimized.backends.cpp.cppBinaryFuzzyARTMAP",
        [os.path.join("artlib", "optimized", "backends", "cpp",
                      "cppBinaryFuzzyARTMAP.cpp")],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        "artlib.optimized.backends.cpp.cppFuzzyARTMAP",
        [os.path.join("artlib", "optimized", "backends", "cpp",
                      "cppFuzzyARTMAP.cpp")],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        "artlib.optimized.backends.cpp.cppHypersphereARTMAP",
        [os.path.join("artlib", "optimized", "backends", "cpp",
                      "cppHypersphereARTMAP.cpp")],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        "artlib.optimized.backends.cpp.cppGaussianARTMAP",
        [os.path.join("artlib", "optimized", "backends", "cpp",
                      "cppGaussianARTMAP.cpp")],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
]

setup(
    name="artlib",
    version="0.1.7",
    packages=find_packages(),  # This ensures artlib, artlib.common, etc. are all included
    ext_modules=ext_modules,
)
