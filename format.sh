#!/usr/bin/env bash

set -euo pipefail

# 设置 clang-format 命令路径，默认为系统 clang-format
CLANG_FORMAT="${CLANG_FORMAT:-clang-format}"

# 定义要格式化的文件后缀列表（支持通过环境变量传入）
SUFFIXES="${SUFFIXES:-.h .cc .cpp .hpp .cxx .hxx .C}"

# 设置根目录（默认当前目录）
ROOT_DIR="."

# 设置并行任务数（可通过环境变量覆盖）
PARALLEL_JOBS="${PARALLEL_JOBS:-$(nproc)}"

# 检查是否安装了 clang-format
if ! command -v "$CLANG_FORMAT" &> /dev/null; then
    echo "Error: clang-format is not installed."
    exit 1
fi

# 输出版本信息
echo "Using $($CLANG_FORMAT --version)"

# 检查是否安装了 parallel
USE_PARALLEL=true
if ! command -v parallel &> /dev/null; then
    echo "Warning: GNU parallel is not installed. Falling back to sequential processing."
    echo "For faster execution, consider installing GNU parallel:"
    echo "  Ubuntu/Debian: sudo apt-get install parallel"
    echo "  macOS: brew install parallel"
    USE_PARALLEL=false
fi

# 将 SUFFIXES 字符串转换为 find 表达式
IFS=' ' read -r -a suffix_array <<< "$SUFFIXES"
find_expr=()
for suffix in "${suffix_array[@]}"; do
    find_expr+=(-name "*$suffix")
    find_expr+=(-o)
done
unset 'find_expr[${#find_expr[@]}-1]' # 移除最后一个多余的 -o

# 格式化单个文件的函数
format_file() {
    local file="$1"
    if git check-ignore -q "$file"; then
        echo "Skipping ignored file by .gitignore: $file"
    else
        echo "Formatting: $file"
        "$CLANG_FORMAT" -i -style=file "$file"
    fi
}

# 导出函数和环境变量以便在子shell中使用
export -f format_file
export CLANG_FORMAT

# 使用 find 查找文件，并排除 third-party 目录，根据是否安装 parallel 选择处理方式
echo "Formatting files..."
if $USE_PARALLEL; then
    echo "Running with $PARALLEL_JOBS parallel jobs"
    find "${ROOT_DIR}" \
        -path "${ROOT_DIR}/third-party" -prune -o \
        -type f \( "${find_expr[@]}" \) -print0 | \
        parallel -0 -j "$PARALLEL_JOBS" format_file
else
    # 串行处理版本
    find "${ROOT_DIR}" \
        -path "${ROOT_DIR}/third-party" -prune -o \
        -type f \( "${find_expr[@]}" \) -print0 | \
        while IFS= read -r -d '' file; do
            format_file "$file"
        done
fi

echo "Formatting completed."