cmake_minimum_required(VERSION 3.21)

# Project name and languages
project(Simulation LANGUAGES CXX CUDA)

# ======================
# Compiler Standards
# ======================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ======================
# CUDA Toolkit
# ======================
find_package(CUDAToolkit REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)

# ======================
# Sources
# ======================
# Adjust these to match your folder structure
file(GLOB_RECURSE SIM_SRC
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/source/*.cu"
)

add_executable(simulation ${SIM_SRC})

# ======================
# CUDA Properties
# ======================
# Enable separable compilation if you have multiple .cu files calling __device__ functions
set_target_properties(simulation PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 86    # adjust to your GPU (e.g. 75 for Turing, 86 for Ampere, 89 for Ada)
)

# ======================
# Includes and Libraries
# ======================
target_include_directories(simulation
    PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(simulation
    PRIVATE
        CUDA::cudart
        CUDA::cublas
        CUDA::curand
)
target_link_libraries(simulation PRIVATE GLEW::GLEW GLUT::GLUT)

# ======================
# Debug flags
# ======================
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Enabling debug flags")
    target_compile_options(simulation PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/Zi /Od>
        $<$<COMPILE_LANGUAGE:CUDA>:-G>
    )
endif()
