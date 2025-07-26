# cmake/CopyDependencies.cmake
# 是否启用 CUDA 的 DLL 复制？默认为 ON（即：独立使用）
option(COPY_CUDA_DLLS "Whether to copy CUDA runtime DLLs" ON)
set(CUDA_BIN_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin" CACHE PATH "CUDA bin directory")

# 设置要复制的目标目录（由外部传入）
message(STATUS "Copying dependencies to: ${DEST_DIR}")

# 设置额外依赖搜索路径（例如：TensorRT、vcpkg）
set(DEPENDENCY_DIRS
    ${TENSORRT_RUNTIME}        # TensorRT 的 bin 目录
    ${VCPKG_INSTALLED_DIR}/x64-windows/bin  # vcpkg 安装目录下的 bin
)

# 如果启用 CUDA，则加入 CUDA 搜索路径
if (COPY_CUDA_DLLS)
    list(APPEND DEPENDENCY_DIRS ${CUDA_BIN_DIR})
endif()

# 收集运行时依赖（优化过滤规则）
file(GET_RUNTIME_DEPENDENCIES
    EXECUTABLES ${BINARY_DIR}
    RESOLVED_DEPENDENCIES_VAR resolved_deps
    UNRESOLVED_DEPENDENCIES_VAR unresolved_deps
    CONFLICTING_DEPENDENCIES_PREFIX conflicting_deps
    DIRECTORIES ${DEPENDENCY_DIRS}
    
    # 排除系统核心 DLL（先大范围排除）
    PRE_EXCLUDE_REGEXES
        "^api-ms-"               # Windows API 集
        "^ext-ms-"               # Windows 扩展组件
        "^(kernel32|user32|gdi32|advapi32|shell32|ole32|ntdll|combase|msvcrt)\\.dll$"
        "^(ws2_32|iphlpapi|crypt32|dnsapi|wldap32|rpcrt4|sechost)\\.dll$"
        "^(dwmapi|uxtheme|dwrite|imm32|winmm|version|powrprof)\\.dll$"
    
    # 包含必要的依赖（第三方库+运行时）
    POST_INCLUDE_REGEXES
        "^(cudart|cublas|cudnn|nvinfer|nvonnxparser|opencv|zlib|icu).*\\.dll$"  # 第三方库
        "^(vcruntime|ucrt|concrt|msvcp|msvcr)\\d*\\.dll$"                          # VC 运行时
    
    # 最终排除可能残留的系统 DLL（兜底）
    POST_EXCLUDE_REGEXES
        "^.*\\.drv$"              # 驱动文件
        "^(dxgi|d3d\\d+|dcomp|gdiplus|windowscodecs)\\.dll$"  # DirectX/图形系统库
        "^(setupapi|cfgmgr32|devobj|devmgr)\\.dll$"            # 设备管理相关
        "^(wtsapi32|userenv|winsta|cryptnet|cryptui)\\.dll$"  # 系统服务相关
    
    VERBOSE  # 启用详细输出，便于调试
)

# 创建目标目录（如果不存在）
file(MAKE_DIRECTORY "${DEST_DIR}")

# 复制已解析的依赖项到目标目录
foreach(dep IN LISTS resolved_deps)
    get_filename_component(filename ${dep} NAME)
    message(STATUS "Copying dependency: ${filename}")
    
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${dep} ${DEST_DIR}
        RESULT_VARIABLE copy_result
    )
    
    if(NOT copy_result EQUAL 0)
        message(WARNING "Failed to copy dependency: ${dep}")
    endif()
endforeach()

# 输出未解析的依赖项（调试用）
if(unresolved_deps)
    # 过滤掉已知的系统DLL未找到警告
    set(filtered_unresolved_deps "")
    foreach(dep IN LISTS unresolved_deps)
        if(NOT dep MATCHES "^api-ms-" AND 
           NOT dep MATCHES "^ext-ms-" AND 
           NOT dep MATCHES "^kernel32\\.dll$")
            list(APPEND filtered_unresolved_deps ${dep})
        endif()
    endforeach()
    
    if(filtered_unresolved_deps)
        message(WARNING "真正未解决的依赖: ${filtered_unresolved_deps}")
    else()
        message(STATUS "未解决的依赖都是系统DLL，可忽略")
    endif()
endif()

# 输出冲突的依赖项（调试用）
if(conflicting_deps)
    message(WARNING "检测到依赖冲突: ${conflicting_deps}")
    # 可添加冲突处理逻辑，例如选择特定版本
endif()