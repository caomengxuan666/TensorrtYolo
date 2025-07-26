# 复制model
file(COPY ${CMAKE_SOURCE_DIR}/model/best.onnx DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/model)

# 复制image
file(COPY ${CMAKE_SOURCE_DIR}/images DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_executable(Infersample infer.cpp 
${CMAKE_SOURCE_DIR}/TensorrtYolo/src/TensorrtYoloDetectorAPI.cpp
${CMAKE_SOURCE_DIR}/TensorrtYolo/include/TensorrtYoloDetectorAPI.h
)

target_include_directories(Infersample
    PRIVATE
        ${CMAKE_SOURCE_DIR}/TensorrtYolo/include
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(Infersample PUBLIC 
        ${OpenCV_LIBS}
        ${CUDART_LIB}
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
)

# 安装可执行文件
install(TARGETS Infersample
    RUNTIME DESTINATION bin)

# 安装 model/
install(DIRECTORY ${CMAKE_SOURCE_DIR}/model/
    DESTINATION model
    FILES_MATCHING PATTERN "*.engine")

# 安装 images/
install(DIRECTORY ${CMAKE_SOURCE_DIR}/images/
    DESTINATION images
    FILES_MATCHING PATTERN "*.jpg" PATTERN "*.png")
