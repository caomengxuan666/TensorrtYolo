# CMakeList.txt: TensorrtYolo 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
add_library(TensorrtYolo SHARED "include/TensorrtYoloDetectorAPI.h" "src/TensorrtYoloDetectorAPI.cpp" "include/Detection.h")

target_compile_definitions(TensorrtYolo PRIVATE TENSORRTYOLO_EXPORTS)

target_include_directories(TensorrtYolo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

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
