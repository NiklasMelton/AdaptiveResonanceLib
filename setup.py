from setuptools import setup, find_packages, Extension
import pybind11
import os
import sys

cpp_source = os.path.join("artlib", "cpp_optimized", "BinaryFuzzyARTMAP.cpp")

extra_compile_args = ["/std:c++17"] if sys.platform == "win32" else ["-std=c++17"]

ext_modules = [
    Extension(
        "artlib.cpp_optimized.BinaryFuzzyARTMAP",
        [cpp_source],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
]

setup(
    name="artlib",
    version="0.1.3",
    packages=find_packages(include=["artlib", "artlib.cpp_optimized"]),
    package_data={
        "artlib.cpp_optimized": ["*.pyd", "*.so"]
    },  # Ensure compiled files are included
    include_package_data=True,
    ext_modules=ext_modules,
    zip_safe=False,
)
