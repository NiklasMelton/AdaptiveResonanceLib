cmake_minimum_required(VERSION 3.15...4.3)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

set(ARTLIB_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/artlib/optimized/backends/cpp")
set(ARTLIB_CPP_MODULES
    cppBinaryFuzzyARTMAP
    cppFuzzyARTMAP
    cppHypersphereARTMAP
    cppGaussianARTMAP
    cppART1MAP
    fracsort
    cppFuzzyART
    cppBinaryFuzzyART
    cppART1)

foreach(module IN LISTS ARTLIB_CPP_MODULES)
  pybind11_add_module(${module} "${ARTLIB_CPP_DIR}/${module}.cpp")
  target_compile_features(${module} PRIVATE cxx_std_17)
  target_include_directories(${module} PRIVATE "${ARTLIB_CPP_DIR}")
  install(
    TARGETS ${module}
    LIBRARY DESTINATION artlib/optimized/backends/cpp
    RUNTIME DESTINATION artlib/optimized/backends/cpp)
endforeach()
