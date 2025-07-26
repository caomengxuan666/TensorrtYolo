add_library(TensorrtYolo SHARED
    "include/TensorrtYoloDetectorAPI.h"
    "src/TensorrtYoloDetectorAPI.cpp"
    "include/Detection.h"
)

target_compile_definitions(TensorrtYolo PRIVATE TENSORRTYOLO_EXPORTS)

target_include_directories(TensorrtYolo PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(TensorrtYolo PRIVATE
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
    set_property(TARGET TensorrtYolo PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# 安装动态库到 bin/
install(TARGETS TensorrtYolo
    EXPORT TensorrtYoloTargets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION bin
    ARCHIVE DESTINATION lib
)

# 安装公共头文件
install(DIRECTORY include/ DESTINATION include)
if(WIN32 AND NOT CYGWIN)
    add_custom_command(TARGET TensorrtYolo POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "Collecting runtime dependencies..."
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_INSTALL_PREFIX}/bin"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:TensorrtYolo>
            "${CMAKE_INSTALL_PREFIX}/bin"

        # 收集所有依赖 DLL
        COMMAND ${CMAKE_COMMAND} -D "BINARY_DIR=$<TARGET_FILE:TensorrtYolo>" -D "DEST_DIR=${CMAKE_INSTALL_PREFIX}/bin" -P ${CMAKE_SOURCE_DIR}/cmake/CopyDependencies.cmake
        # 复制cudart
    )
endif()