# CMakeList.txt: 顶层 CMake 项目文件，在此处执行全局配置
# 并包含子项目。
#
cmake_minimum_required (VERSION 3.8)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX STANDARD ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# 如果支持，请为 MSVC 编译器启用热重载。
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("TensorrtYolo")

set(VCPKG_ROOT D:/vcpkg)
set(CMAKE_TOOLCHAIN_FILE "${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
set(CMAKE_PREFIX_PATH "${VCPKG_ROOT}/installed/x64-windows/share" ${CMAKE_PREFIX_PATH})
set(CUDA_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9")
set(TENSORRT_INCLUDE ${CUDA_DIR}/include)
set(TENSORRT_LIB ${CUDA_DIR}/lib/x64)
message(STATUS ${TENSORRT_LIB})
set(TENSORRT_RUNTIME ${CUDA_DIR}/bin)

include_directories(${TENSORRT_INCLUDE})
link_directories(${TENSORRT_LIB})

find_package(OpenCV CONFIG REQUIRED)

find_library(CUDART_LIB
    NAMES cudart cudart64_120
    PATHS ${TENSORRT_LIB}
)
if (NOT CUDART_LIB)
    message(FATAL_ERROR "Could NOT find cudart.lib in ${TENSORRT_LIB}")
endif()

find_library(NVINFER_LIB NAMES nvinfer_10 PATHS ${TENSORRT_LIB})

if (NOT NVINFER_LIB)
    message(FATAL_ERROR "Could NOT find nvinfer_10.lib in ${TENSORRT_ROOT_DIR}")
endif()


if (MSVC)
add_compile_options(/arch:AVX2)
message(STATUS "Using MSVC compiler with AVX2 support")
else ()
add_compile_options(-mavx2)
message(STATUS "Using non-MSVC compiler with AVX2 support")
endif ()

# 包含子项目。
add_subdirectory ("TensorrtYolo")
add_subdirectory ("benchmark")
add_subdirectory("samples")