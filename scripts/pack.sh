#!/bin/bash
# 保存为 pack_libs.sh 并运行
mkdir -p libs
echo "正在打包依赖库..."
# 提取 sigmo_mpi 的依赖，重点是 libsy* (SYCL) 和 libmpi* (MPI)
ldd build/sigmo_mpi | grep "intel" | awk '{print $3}' | sort | uniq | xargs -I {} cp -L {} libs/
ldd build/sigmo_mpi | grep "libmpi" | awk '{print $3}' | sort | uniq | xargs -I {} cp -L {} libs/ 2>/dev/null
ldd build/sigmo2_mpi_dynamic | grep "intel" | awk '{print $3}' | sort | uniq | xargs -I {} cp -L {} libs/
ldd build/sigmo2_mpi_dynamic | grep "libmpi" | awk '{print $3}' | sort | uniq | xargs -I {} cp -L {} libs/ 2>/dev/null
echo "打包完成，所有库都在 libs/ 目录下。"