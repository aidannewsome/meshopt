cmake_minimum_required(VERSION 3.18...3.31)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT})
find_package(nanobind CONFIG REQUIRED)

# The binding and meshoptimizer's own sources, the files the encoder and decoder need, in one module.
nanobind_add_module(_core STABLE_ABI FREE_THREADED NB_STATIC LTO
    src/meshopt.cpp
    vendor/meshoptimizer/allocator.cpp
    vendor/meshoptimizer/vertexcodec.cpp
    vendor/meshoptimizer/indexcodec.cpp
    vendor/meshoptimizer/vertexfilter.cpp
    vendor/meshoptimizer/vcacheoptimizer.cpp
    vendor/meshoptimizer/vfetchoptimizer.cpp
    vendor/meshoptimizer/spatialorder.cpp
)

install(TARGETS _core LIBRARY DESTINATION meshopt)
