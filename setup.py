# setup.py
import os
import sys

import pybind11
from setuptools import setup, Extension, find_packages

extra_compile_args = ["/std:c++17"] if sys.platform == "win32" else ["-std=c++17"]

cpp_dir = os.path.join("artlib", "optimized", "backends", "cpp")
cpp_include_dirs = [pybind11.get_include(), cpp_dir]


def cpp_extension(module_name, source_name):
    return Extension(
        module_name,
        [os.path.join(cpp_dir, source_name)],
        include_dirs=cpp_include_dirs,
        language="c++",
        extra_compile_args=extra_compile_args,
    )

ext_modules = [
    cpp_extension(
        "artlib.optimized.backends.cpp.cppBinaryFuzzyARTMAP",
        "cppBinaryFuzzyARTMAP.cpp",
    ),
    cpp_extension(
        "artlib.optimized.backends.cpp.cppFuzzyARTMAP",
        "cppFuzzyARTMAP.cpp",
    ),
    cpp_extension(
        "artlib.optimized.backends.cpp.cppHypersphereARTMAP",
        "cppHypersphereARTMAP.cpp",
    ),
    cpp_extension(
        "artlib.optimized.backends.cpp.cppGaussianARTMAP",
        "cppGaussianARTMAP.cpp",
    ),
    cpp_extension(
        "artlib.optimized.backends.cpp.cppART1MAP",
        "cppART1MAP.cpp",
    ),
    cpp_extension(
        "artlib.optimized.backends.cpp.fracsort",
        "fracsort.cpp",
    ),
    cpp_extension(
        "artlib.optimized.backends.cpp.cppFuzzyART",
        "cppFuzzyART.cpp",
    ),
    cpp_extension(
        "artlib.optimized.backends.cpp.cppBinaryFuzzyART",
        "cppBinaryFuzzyART.cpp",
    ),
    cpp_extension(
        "artlib.optimized.backends.cpp.cppART1",
        "cppART1.cpp",
    ),
]

setup(
    name="artlib",
    version="0.1.9",
    packages=find_packages(),  # This all are included
    include_package_data=True,
    package_data={"artlib.optimized.backends.cpp": ["*.cpp", "*.hpp"]},
    ext_modules=ext_modules,
)
