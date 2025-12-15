#!/bin/bash
# Copyright (c) 2025 University of Salerno
# SPDX-License-Identifier: Apache-2.0

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# 默认配置
sigmo_arch="nvidia_gpu_sm_70"
sigmo_compiler=""

help()
{
  echo "Usage: ./1_build.sh 
      [ --sigmo-arch= ]   The target architecture for BMS2 (Default: nvidia_gpu_sm_70);
      [ --sigmo-compiler= ] The compiler for BMS2 (Default: auto-detect icpx);
      [ -h | --help ]       Print this help message and exit."
}

# 解析参数
while [[ $# -gt 0 ]]; do
  case "$1" in
    --sigmo-arch=*)
      sigmo_arch="${1#*=}"
      shift
      ;;
    --sigmo-compiler=*)
      sigmo_compiler="${1#*=}"
      shift
      ;;
    -h | --help)
      help
      exit 0
      ;;
    *)
      echo "Invalid argument: $1"
      help
      exit 1
      ;;
  esac
done

# 自动检测 Intel icpx 编译器
if [ -z "$sigmo_compiler" ]; then
  if command -v icpx &> /dev/null; then
    sigmo_compiler=$(which icpx)
  else
    echo "[!] Intel oneAPI compiler (icpx) not found."
    echo "    Please load the oneAPI environment (e.g., source /opt/intel/oneapi/setvars.sh)"
    echo "    or specify the compiler using --sigmo-compiler=<path_to_icpx>"
    exit 1
  fi
fi

echo "[*] Starting BMS2 build process..."
echo "[*] Architecture: $sigmo_arch"
echo "[*] Compiler: $sigmo_compiler"

# 创建并进入构建目录
mkdir -p "$SCRIPT_DIR/build"
cd "$SCRIPT_DIR/build" || exit 1

# 执行 CMake 配置和构建
cmake ../library \
  -DCMAKE_CXX_COMPILER="$sigmo_compiler" \
  -DSIGMO_TARGET_ARCHITECTURE="$sigmo_arch" \
  -DSIGMO_ENABLE_TEST=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCXXOPTS_BUILD_EXAMPLES=OFF \
  -DCXXOPTS_BUILD_TESTS=OFF \
  -DCXXOPTS_ENABLE_INSTALL=OFF

if [ $? -eq 0 ]; then
  echo "[*] Configuring CMake successful. Compiling..."
  cmake --build . -j$(nproc)
  echo "[*] BMS2 build completed successfully."
else
  echo "[!] CMake configuration failed."
  exit 1
fi