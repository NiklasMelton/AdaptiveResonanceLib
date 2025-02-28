# setup.py
import sys
import os
import pybind11
from setuptools import setup, Extension, find_packages

extra_compile_args = ["/std:c++17"] if sys.platform == "win32" else ["-std=c++17"]

ext_modules = [
    Extension(
        "artlib.cpp_optimized.cppBinaryFuzzyARTMAP",
        [os.path.join("artlib", "cpp_optimized", "cppBinaryFuzzyARTMAP.cpp")],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=extra_compile_args,
    )
]

setup(
    name="artlib",
    version="0.1.3",
    packages=find_packages(),  # This ensures artlib, artlib.common, etc. are all included
    ext_modules=ext_modules,
)
