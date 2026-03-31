cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(rxmesh_torch_ops LANGUAGES CXX CUDA)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
set(CMAKE_BUILD_TYPE Debug)
if(NOT Torch_DIR)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import torch, os; print(torch.utils.cmake_prefix_path())"
        OUTPUT_VARIABLE TORCH_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(TORCH_PREFIX)
        list(APPEND CMAKE_PREFIX_PATH "${TORCH_PREFIX}")
        set(Torch_DIR "${TORCH_PREFIX}/share/cmake/Torch" CACHE PATH "Path to TorchConfig.cmake" FORCE)
    endif()
endif()

find_package(Torch REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 86) # adjust this to your GPU architecture

set(EIGEN3_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../rxmesh/third_party/eigen")
# As EIGEN may download failed and cause build failure, we can use the system-wide EIGEN if available
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc
    ${TORCH_INCLUDE_DIRS}
    ${RXMESH_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    ../rxmesh/include
    ${EIGEN3_INCLUDE_DIR}
    ../rxmesh/third_party/spdlog/include
    ../build/_deps/spdlog-src/include
)

set(SOURCES
    csrc/bindings.cpp
    csrc/rxmesh_wrapper.cu
    csrc/topology.cu
    csrc/geometry.cu
)

add_library(rxmesh_torch_ops SHARED ${SOURCES})

set_target_properties(rxmesh_torch_ops PROPERTIES
    PREFIX ""  # remove 'lib' prefix
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_link_libraries(rxmesh_torch_ops PRIVATE
    ${TORCH_LIBRARIES}
    ${CUDA_LIBRARIES}
    CUDA::cudart
    CUDA::cuda_driver
)

if(TARGET RXMesh)
    target_link_libraries(rxmesh_torch_ops PRIVATE RXMesh)
endif()

set_target_properties(rxmesh_torch_ops PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
)

target_compile_options(rxmesh_torch_ops PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --expt-extended-lambda
        # -G                  # debug symbols
        # -lineinfo
        # -Xcompiler=-O0
        -use_fast_math
    >
)

target_compile_definitions(rxmesh_torch_ops PRIVATE
    TORCH_EXTENSION_NAME=rxmesh_torch_ops
)

find_library(TORCH_PYTHON_LIB
    NAMES torch_python
    PATHS "${TORCH_INSTALL_PREFIX}/lib" "${TORCH_INSTALL_PREFIX}/lib64"
    NO_DEFAULT_PATH
)

if(TORCH_PYTHON_LIB)
    target_link_libraries(rxmesh_torch_ops PRIVATE "${TORCH_PYTHON_LIB}")
else()
    message(FATAL_ERROR "Cannot find torch_python (libtorch_python.so). Please locate it under torch/lib or torch/lib64.")
endif()


# add_executable(rxmesh_torch_test debug.cu)
# target_link_libraries(rxmesh_torch_test PRIVATE 
#     cuda 
#     ${TORCH_LIBRARIES}
#     rxmesh_torch_ops 
#     Python3::Python
    
#     )