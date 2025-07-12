# CMakeList.txt: TensorrtYolo 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
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

# 复制model
file(COPY ${CMAKE_SOURCE_DIR}/model/best.onnx DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/model)

# 复制image
file(COPY ${CMAKE_SOURCE_DIR}/images DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# 将源代码添加到此项目的可执行文件。
add_executable (TensorrtYolo "TensorrtYolo.cpp" "TensorrtYolo.h")

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        kernel32
        user32
        gdi32
        winspool
        comdlg32
        advapi32
        shell32
        ole32
        oleaut32
        uuid
        odbc32
        odbccp32
        nvinfer_10
        nvinfer_plugin_10
        nvonnxparser_10
        ${CUDART_LIB}
        ${OpenCV_LIBS}
)

# 设置 Windows 下静态运行时 /MT 或 /MTd（避免与 CRT 冲突）
if(MSVC)
    set_property(TARGET ${PROJECT_NAME} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET TensorrtYolo PROPERTY CXX_STANDARD 20)
endif()
# TODO: 如有需要，请添加测试并安装目标。
