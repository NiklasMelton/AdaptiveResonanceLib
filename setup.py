# setup.py
import os
import sys
import pybind11
from setuptools import setup, Extension

extra_compile_args = ["/std:c++17"] if sys.platform == "win32" else ["-std=c++17"]

ext_modules = [
    Extension(
        "artlib.cpp_optimized.BinaryFuzzyARTMAP",  # as you have now
        [os.path.join("artlib", "cpp_optimized", "BinaryFuzzyARTMAP.cpp")],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=extra_compile_args,
    )
]

setup(
    name="artlib",
    version="0.1.3",
    packages=["artlib", "artlib.cpp_optimized"],
    ext_modules=ext_modules,
)
