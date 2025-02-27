from setuptools.extension import Extension
import pybind11
import os
import sys

cpp_source = os.path.join("artlib", "cpp_optimized", "BinaryFuzzyARTMAP.cpp")

extra_compile_args = ["/std:c++17"] if sys.platform == "win32" else ["-std=c++17"]

custom_extension = Extension(
    "artlib.cpp_optimized.BinaryFuzzyARTMAP",
    [cpp_source],
    include_dirs=[pybind11.get_include()],
    language="c++",
    extra_compile_args=extra_compile_args,
)


def build(setup_kwargs):
    """This is a callback for Poetry used to hook in our C++ extensions."""
    setup_kwargs.update(
        {
            "ext_modules": [custom_extension],
            "zip_safe": False,
            "include_package_data": True,
        }
    )
