#!/bin/bash
# Copyright (c) 2025 University of Salerno
# SPDX-License-Identifier: Apache-2.0

SCRIPT_DIR=$1
shift

EXPERIMENTS="core,diameter,dataset-scale,gpu-metrics,mpi,cal,join,balance,ablation,partial,memory"
experiments=""
total_iterations=7
zinc_dataset=""

function help() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  -e=, --experiments=<exp1,exp2,...>  Comma-separated list of experiments to run (default: $EXPERIMENTS)"
  echo "  -i=, --iterations=<num>            Number of iterations for each experiment (default: $total_iterations)"
  echo "  --zinc=<path>             Path to the ZINC dataset (required if running mpi experiment)"
  echo "  -H, --help                        Display this help message"
}

# Parsing arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -e=*|--experiments=*)
      experiments="${1#*=}"
      shift
      ;;
    -i=*|--iterations=*)
      total_iterations="${1#*=}"
      shift
      ;;
    --zinc=*)
      zinc_dataset="${1#*=}"
      zinc_dataset=$(realpath "$zinc_dataset")
      shift
      ;;
    -H)
      help
      exit 0
      ;;
    *)
      help
      exit 1
      ;;
  esac
done

# check if experiments is empty and valid
for exp in $(echo $experiments | sed "s/,/ /g")
do
  if [[ $EXPERIMENTS != *"$exp"* ]]
  then
    echo "[!] Invalid experiment: $exp"
    return 1 2>/dev/null
    exit 1
  fi
done
if [ -z "$experiments" ]
then
  experiments=$EXPERIMENTS
fi

start_time=$(date +%s)

if [[ $experiments == *"core"* ]]; then
  echo "Running SIGMo assessment experiments..."
  # 基础输出目录
  OUT_DIR="$SCRIPT_DIR/out/SIGMO/logs/core"
  
  # 定义包含 .dat 文件的目标数据集目录
  DATASET_PATH="/home/hujiayue/antonio-decaro-SIGMo-304d11c/data/SIGMO/dataset"
  # 定义包含 .dat 文件的查询集目录
  QUERY_PATH="/home/hujiayue/antonio-decaro-SIGMo-304d11c/data/SIGMO/queryset_part"

  # --- 外层循环：遍历所有查询集文件 ---
  for query_file in "$QUERY_PATH"/*.dat; do
    # 提取查询文件名（作为第一级文件夹名）
    query_name=$(basename "$query_file" .dat)
    echo "Current Query Set: $query_name"

    # --- 内层循环：遍历所有数据集文件 ---
    for data_file in "$DATASET_PATH"/*.dat; do
      # 提取数据文件名（作为第二级文件夹名）
      data_name=$(basename "$data_file" .dat)
      echo "  -> Processing Dataset: $data_name with Query: $query_name"

      # 构建嵌套的输出路径： origin/查询名/数据名
      CURRENT_ORIGIN_DIR="$OUT_DIR/origin/$query_name/$data_name"
      CURRENT_OURS_DIR="$OUT_DIR/ours/$query_name/$data_name"

      # 创建目录
      mkdir -p "$CURRENT_ORIGIN_DIR"
      mkdir -p "$CURRENT_OURS_DIR"

      for i in $(seq 1 2); do
        echo "    Run $i/2..."

        # 运行 sigmo
        # -Q 使用当前的 $query_file
        # -D 使用当前的 $data_file
        $SCRIPT_DIR/build/sigmo -i 6 -Q "$query_file" -D "$data_file" -p -c query --find-all --max-data-graphs 1000000000 > "$CURRENT_ORIGIN_DIR/sigmo_findall_${i}.log" 2> "$CURRENT_ORIGIN_DIR/err_sigmo_findall_${i}.log"

        # 运行 sigmo2 (ours)
        $SCRIPT_DIR/build/sigmo2 -i 6 -Q "$query_file" -D "$data_file" -p -c query --find-all --max-data-graphs 1000000000 > "$CURRENT_OURS_DIR/sigmo_findall_${i}.log" 2> "$CURRENT_OURS_DIR/err_sigmo_findall_${i}.log"
      done
    done
  done

  # 运行结果分析脚本
  source $SCRIPT_DIR/.venv/bin/activate
  # !重要提示!：
  # 现在目录结构变成了 /origin/query_name/data_name/log_files
  # 请务必检查 output_analyzer.py 是否能递归遍历两层子目录，或者需要修改该 Python 脚本以适应新的路径结构。
  python $SCRIPT_DIR/scripts/utils/output_analyzer.py $OUT_DIR $SCRIPT_DIR/out/SIGMO/sigmo_results.csv
  deactivate
fi

if [[ $experiments == *"cal"* ]]; then
  echo "Running SIGMo assessment experiments..."
  OUT_DIR="$SCRIPT_DIR/out/SIGMO/logs/cal"
  
  # 定义输入数据的绝对路径
  QUERY_DIR="/home/hujiayue/antonio-decaro-SIGMo-304d11c/data/SIGMO/candidateset"
  DATA_DIR="/home/hujiayue/antonio-decaro-SIGMo-304d11c/data/SIGMO/dataset"

  # 遍历 dataset 目录下所有的 .dat 文件 (作为目录结构的第一层：数据集名字)
  for data_file in "$DATA_DIR"/*.dat; do
    # 获取数据集文件名 (例如: data.dat -> data)
    d_name=$(basename "$data_file" .dat)

    # 遍历 candidateset 目录下所有的 .dat 文件 (作为目录结构的第二层：query名字)
    for query_file in "$QUERY_DIR"/*.dat; do
      # 获取 Query 文件名 (例如: candidates.dat -> candidates)
      q_name=$(basename "$query_file" .dat)
      
      # 构建目标存储目录: .../cal/数据集名字/query名字/
      TARGET_DIR="$OUT_DIR/$d_name/$q_name"
      mkdir -p "$TARGET_DIR"

      echo "Running Data: $d_name vs Query: $q_name"
      echo "Logs will be saved to: $TARGET_DIR"

      # 运行原始版本 (origin)
      # 结果保存在 TARGET_DIR/origin.log
      CUDA_VISIBLE_DEVICES=2 $SCRIPT_DIR/build/sigmo_cal \
        -i 6 \
        -Q "$query_file" \
        -D "$data_file" \
        -p -c query --find-all \
        > "$TARGET_DIR/origin.log" \
        2> "$TARGET_DIR/err_origin.log"

      # 运行我们的版本 (ours)
      # 结果保存在 TARGET_DIR/ours.log
      CUDA_VISIBLE_DEVICES=2 $SCRIPT_DIR/build/sigmo2_cal \
        -i 6 \
        -Q "$query_file" \
        -D "$data_file" \
        -p -c query --find-all \
        > "$TARGET_DIR/ours.log" \
        2> "$TARGET_DIR/err_ours.log"
        
    done
  done

  source $SCRIPT_DIR/.venv/bin/activate
  # 注意：由于目录结构发生了巨大变化（变成了嵌套目录），原本的 output_analyzer.py 
  # 很可能无法直接解析新的路径结构，请根据需要检查或修改 python 脚本。
  python $SCRIPT_DIR/scripts/utils/output_analyzer.py $OUT_DIR $SCRIPT_DIR/out/SIGMO/sigmo_results.csv
  deactivate
fi

if [[ $experiments == *"join"* ]]; then
  echo "Running SIGMo assessment experiments (Fixed Query)..."
  OUT_DIR="$SCRIPT_DIR/out/SIGMO/logs/join"
  mkdir -p "$OUT_DIR"
  
  # 1. 修改：固定 Query 文件的绝对路径
  QUERY_FILE="/home/hujiayue/antonio-decaro-SIGMo-304d11c/data/SIGMO/queryset/query5.dat"
  
  # 定义输入数据的目录
  DATA_DIR="/home/hujiayue/antonio-decaro-SIGMo-304d11c/data/SIGMO/dataset"

  # 遍历 dataset 目录下所有的 .dat 文件
  for data_file in "$DATA_DIR"/*.dat; do
    # 获取数据集文件名 (例如: data.dat -> data)
    d_name=$(basename "$data_file" .dat)

    # 2. 修改：移除 Query 循环，直接使用固定的 QUERY_FILE
    # 3. 修改：不再创建子文件夹，直接使用 OUT_DIR
    echo "Running Data: $d_name vs Fixed Query: query1"
    echo "Logs will be saved to: $OUT_DIR"
    
    # 运行我们的版本 (ours)
    # 结果文件名包含数据集名称，以防覆盖: dataset名_ours.log
    CUDA_VISIBLE_DEVICES=2 $SCRIPT_DIR/build/join_cal \
      -i 6 \
      -Q "$QUERY_FILE" \
      -D "$data_file" \
      -p -c query --find-all \
      > "$OUT_DIR/${d_name}_ours.log" \
      2> "$OUT_DIR/${d_name}_err_ours.log"
      
  done

  source $SCRIPT_DIR/.venv/bin/activate
  # 注意：由于现在输出目录是扁平的（没有子文件夹），请确保 output_analyzer.py 能够处理这种情况。
  # 以前是传递根目录让它递归查找，现在所有 log 都在一层目录下。
  python $SCRIPT_DIR/scripts/utils/output_analyzer.py $OUT_DIR $SCRIPT_DIR/out/SIGMO/sigmo_results.csv
  deactivate
fi
if [[ $experiments == *"balance"* ]]; then
  echo "Running balance experiments..."
  
  # 1. 设置基础路径
  # 假设 SCRIPT_DIR 是项目根目录 /home/hujiayue/antonio-decaro-SIGMo-304d11c
  # 数据集路径 (保持你代码中的 dataset_small)
  DATASET_DIR="$SCRIPT_DIR/data/SIGMO/dataset_small"
  
  # [修改] 查询集路径 (改为 queryset_nopart)
  QUERYSET_DIR="$SCRIPT_DIR/data/SIGMO/queryset_nopart"
  
  OUT_ROOT="$SCRIPT_DIR/out/SIGMO/logs/balance"

  # 2. 外层循环：遍历数据集目录下的所有 .dat 文件
  for DATA_FILE in "$DATASET_DIR"/*.dat; do
    # 检查文件是否存在
    [ -e "$DATA_FILE" ] || continue

    # 获取数据集的名字 (例如 "yeast")
    DATASET_NAME=$(basename "$DATA_FILE" .dat)

    # 3. [新增] 内层循环：遍历查询集目录下的所有 .dat 文件
    for QUERY_FILE in "$QUERYSET_DIR"/*.dat; do
      # 检查文件是否存在
      [ -e "$QUERY_FILE" ] || continue

      # 获取查询集的名字 (例如 "query_dense")
      QUERYSET_NAME=$(basename "$QUERY_FILE" .dat)

      # 4. 构建输出目录路径: .../balance/数据集名/查询集名
      CURRENT_OUT_DIR="$OUT_ROOT/$DATASET_NAME/$QUERYSET_NAME"
      mkdir -p "$CURRENT_OUT_DIR"

      echo "------------------------------------------------"
      echo "Processing Dataset:  $DATASET_NAME"
      echo "Processing Queryset: $QUERYSET_NAME"
      echo "Output Dir:          $CURRENT_OUT_DIR"

      # 5. 执行原始算法 (origin_load_cal)
      # 使用当前的 $QUERY_FILE 和 $DATA_FILE
      $SCRIPT_DIR/build/origin_load_cal \
        -i 6 -c query \
        -Q "$QUERY_FILE" \
        -D "$DATA_FILE" \
        --find-all --max-data-graphs 1000000000 \
        > "$CURRENT_OUT_DIR/sigmo.log" \
        2> "$CURRENT_OUT_DIR/err_sigmo_origin.log"

      # 6. 执行优化算法 (load_cal)
      # 使用当前的 $QUERY_FILE 和 $DATA_FILE
      $SCRIPT_DIR/build/load_cal \
        -i 6 -c query \
        -Q "$QUERY_FILE" \
        -D "$DATA_FILE" \
        --find-all --max-data-graphs 1000000000 \
        > "$CURRENT_OUT_DIR/our_sigmo.log" \
        2> "$CURRENT_OUT_DIR/err_sigmo_our.log"
        
    done # 结束内层循环 (Query)
  done # 结束外层循环 (Dataset)

  echo "All balance experiments finished."
fi
if [[ $experiments == *"ablation"* ]]; then
  echo "Running ablation experiments..."
  
  # 1. 设置基础路径
  # 假设 SCRIPT_DIR 是项目根目录
  DATASET_DIR="$SCRIPT_DIR/data/SIGMO/dataset_small"
  QUERYSET_DIR="$SCRIPT_DIR/data/SIGMO/queryset_nopart"
  
  # 所有日志直接输出到这个目录，不再创建子文件夹
  OUT_ROOT="$SCRIPT_DIR/out/SIGMO/logs/ablation"
  mkdir -p "$OUT_ROOT"

  # 2. 外层循环：遍历数据集
  for DATA_FILE in "$DATASET_DIR"/*.dat; do
    # 检查文件是否存在
    [ -e "$DATA_FILE" ] || continue

    # 获取数据集的名字 (例如 "yeast")
    DATASET_NAME=$(basename "$DATA_FILE" .dat)

    # 3. 内层循环：遍历查询集
    for QUERY_FILE in "$QUERYSET_DIR"/*.dat; do
      # 检查文件是否存在
      [ -e "$QUERY_FILE" ] || continue

      # 获取查询集的名字 (例如 "query_dense")
      QUERYSET_NAME=$(basename "$QUERY_FILE" .dat)

      # 4. 构建日志文件名：数据集名_查询集名.log
      LOG_FILE="$OUT_ROOT/${DATASET_NAME}_${QUERYSET_NAME}.log"
      ERR_FILE="$OUT_ROOT/${DATASET_NAME}_${QUERYSET_NAME}_err.log"

      echo "------------------------------------------------"
      echo "Processing Dataset:  $DATASET_NAME"
      echo "Processing Queryset: $QUERYSET_NAME"
      echo "Log File:            $LOG_FILE"

      # 执行命令，直接重定向到命名的日志文件
      $SCRIPT_DIR/build/ablation \
        -i 6 -c query \
        -Q "$QUERY_FILE" \
        -D "$DATA_FILE" \
        --find-all --max-data-graphs 1000000000 \
        > "$LOG_FILE"

    done # 结束内层循环 (Query)
  done # 结束外层循环 (Dataset)

  echo "All ablation experiments finished."
fi

if [[ $experiments == *"dataset-scale"* ]]; then
  echo "Running Dataset Scaling experiments..."
  OUT_DIR="$SCRIPT_DIR/out/SIGMO/logs/dataset_scale"
  mkdir -p $OUT_DIR
  for k in {1..25}
  do
    printf "Dataset scale %i\n" $k
    rm -f $OUT_DIR/logs_$k.log
    rm -f $OUT_DIR/logs_findall_$k.log
    rm -f $OUT_DIR/err_$k.log
    rm -f $OUT_DIR/err_findall_$k.log
    for i in {1..5}
    do
      printf "Iteration %i/%s\n" $i 5
      $SCRIPT_DIR/build/sigmo2 -i 5 -c query -Q $SCRIPT_DIR/data/SIGMO/query_nowildcards.dat -D $SCRIPT_DIR/data/SIGMO/data.dat --max-data-graphs 1000000000 --find-all --mul-data=$k --skip-candidates-analysis >> $OUT_DIR/logs_findall_$k.log 2>> $OUT_DIR/err_findall_$k.log
      $SCRIPT_DIR/build/sigmo2 -i 5 -c query -Q $SCRIPT_DIR/data/SIGMO/query_nowildcards.dat -D $SCRIPT_DIR/data/SIGMO/data.dat --max-data-graphs 1000000000 --mul-data=$k --skip-candidates-analysis >> $OUT_DIR/logs_$k.log 2>> $OUT_DIR/err_findall_$k.log
    done
  done
fi

if [[ $experiments == *"diameter"* ]]; then
  echo "Running diameter experiments..."
  OUT_DIR="$SCRIPT_DIR/out/SIGMO/logs/diameter"
  mkdir -p $OUT_DIR

  # list all the files in $SCIRPT_DIR/data/SIGMO/query/*.dat
  query_files=($SCRIPT_DIR/data/SIGMO/query/*.dat)
  

  for query_file in "${query_files[@]}"; do
    query_name=$(basename "$query_file" .dat)
    query_name=$(echo "$query_name" | sed 's/^query_//')
    for i in $(seq 0 $total_iterations); do
      printf "Query %s - Iteration %i/%s\n" "$query_name" $i $total_iterations
      $SCRIPT_DIR/build/sigmo -i $i -c query -Q "$query_file" -D $SCRIPT_DIR/data/SIGMO/data.dat --find-all --skip-candidates-analysis > $OUT_DIR/sigmo_${query_name}_${i}.log
    done
  done
fi

if [[ $experiments == *"gpu-metrics"* ]]; then
  # 定义基础路径
  BASE_OUT_DIR="$SCRIPT_DIR/out/SIGMO/logs/gpu_metrics"
  echo "Running GPU metrics experiments..."
  # ==========================================
  # Part 1: SIGMO (Original)
  # Output: .../gpu_metrics/sigmo/
  # ==========================================
  #echo "  > Testing SIGMO..."
  #OUT_DIR_SIGMO="$BASE_OUT_DIR/sigmo"
  #mkdir -p $OUT_DIR_SIGMO

  ## 1. 启动监控
  #dcgmi dmon -e 1003 -d 10 > $OUT_DIR_SIGMO/dcgmi.log 2> $OUT_DIR_SIGMO/err_dcgmi.log &
  #DCGMI_PID=$!

  # 2. 运行 sigmo
  #$SCRIPT_DIR/build/sigmo -i 5 -c query -Q $SCRIPT_DIR/data/SIGMO/query1.dat -D $SCRIPT_DIR/data/SIGMO/data.dat --find-all --join-work-group=64 --skip-candidates-analysis > $OUT_DIR_SIGMO/sigmo.log 2> $OUT_DIR_SIGMO/err_sigmo.log
  
  # 3. 停止监控
  #kill $DCGMI_PID
  #wait $DCGMI_PID 2>/dev/null

  # 4. 运行 NCU 分析
  #ncu --set full -f -o $OUT_DIR_SIGMO/sigmo $SCRIPT_DIR/build/sigmo -i 5 -c query -Q $SCRIPT_DIR/data/SIGMO/query.dat -D $SCRIPT_DIR/data/SIGMO/data.dat --join-work-group=64 --skip-candidates-analysis
  #ncu -i $OUT_DIR_SIGMO/sigmo.ncu-rep --print-details all --csv --print-metric-name name > $OUT_DIR_SIGMO/metrics.csv


  # ==========================================
  # Part 2: SIGMO2 (Ours)
  # Output: .../gpu_metrics/ours/
  # ==========================================
  echo "  > Testing Ours (sigmo2)..."
  OUT_DIR_OURS="$BASE_OUT_DIR/ours"
  mkdir -p $OUT_DIR_OURS

  # 1. 重新启动监控
  dcgmi dmon -e 1003 -d 10 > $OUT_DIR_OURS/dcgmi.log 2> $OUT_DIR_OURS/err_dcgmi.log &
  DCGMI_PID=$!

  # 2. 运行 sigmo2 (Ours)
  $SCRIPT_DIR/build/sigmo2 -i 5 -c query -Q $SCRIPT_DIR/data/SIGMO/query.dat -D $SCRIPT_DIR/data/SIGMO/data.dat --find-all --join-work-group=64 --skip-candidates-analysis > $OUT_DIR_OURS/sigmo2.log 2> $OUT_DIR_OURS/err_sigmo2.log
  
  # 3. 停止监控
  kill $DCGMI_PID
  wait $DCGMI_PID 2>/dev/null

  # 4. 运行 NCU 分析 (sigmo2)
  ncu --set full -f -o $OUT_DIR_OURS/sigmo2 $SCRIPT_DIR/build/sigmo2 -i 5 -c query -Q $SCRIPT_DIR/data/SIGMO/query1.dat -D $SCRIPT_DIR/data/SIGMO/data.dat --join-work-group=64 --skip-candidates-analysis
  ncu -i $OUT_DIR_OURS/sigmo2.ncu-rep --print-details all --csv --print-metric-name name > $OUT_DIR_OURS/metrics.csv

  echo "All GPU metrics experiments finished."
fi
if [[ $experiments == *"memory"* ]]; then
  # 定义基础路径
  BASE_OUT_DIR="$SCRIPT_DIR/out/SIGMO/logs/memory"
  echo "Running memory experiments..."
  OUT_DIR_OURS="$BASE_OUT_DIR"
  mkdir -p $OUT_DIR_OURS
  # 2. 运行 sigmo2 (Ours)
  CUDA_VISIBLE_DEVICES=7 $SCRIPT_DIR/build/memory -i 6 -c query -Q $SCRIPT_DIR/data/SIGMO/queryset/query.dat -D $SCRIPT_DIR/data/zinc.dat --max-data-graphs 1000000000 --find-all > $OUT_DIR_OURS/memory.log 2> $OUT_DIR_OURS/err_sigmo2.log
  echo "All GPU metrics experiments finished."
fi

if [[ $experiments == *"mpi"* ]]; then
  if [ -z "$zinc_dataset" ]; then
    echo "[!] ZINC dataset path is required for MPI experiments. Please provide it using --zinc=<path>."
    exit 1
  fi

  # [配置] 查询图所在目录
  QUERY_DIR="/home/hujiayue/antonio-decaro-SIGMo-304d11c/data/SIGMO/queryset_nopart"

  # 检查查询目录是否存在
  if [ ! -d "$QUERY_DIR" ]; then
      echo "错误: 查询目录不存在 $QUERY_DIR"
      exit 1
  fi

  echo "Running MPI experiments..."
  # 基础日志目录
  BASE_OUT_DIR="$SCRIPT_DIR/out/SIGMO/logs/mpi"
  mkdir -p "$BASE_OUT_DIR"

  # [配置] 物理GPU数量 (根据你的机器配置，这里设为8)
  PHYSICAL_GPUS=8
  
  # 计算数据集的总行数
  echo "Counting dataset size..."
  TOTAL_GRAPHS=$(wc -l < "$zinc_dataset")
  echo "Total graphs in dataset: $TOTAL_GRAPHS"

  # 定义要模拟的 GPU 规模列表
  SIMULATED_SCALES=(4 8 16 32 64 128 256)

  # --- 外层循环：遍历不同的模拟规模 ---
  for SIM_NP in "${SIMULATED_SCALES[@]}"; do
    
    # ---------------------------------------------------------
    # [计算逻辑] 判断当前运行的实际进程数 (CURRENT_RUN_NP)
    # ---------------------------------------------------------
    if [ "$SIM_NP" -lt "$PHYSICAL_GPUS" ]; then
        # 正常运行
        CURRENT_RUN_NP=$SIM_NP
        echo "========================================================"
        echo "  -> Normal Run: $SIM_NP GPUs"
        echo "========================================================"
    else
        # 模拟运行
        CURRENT_RUN_NP=$PHYSICAL_GPUS
        echo "========================================================"
        echo "  -> Simulating Cluster Size: $SIM_NP GPUs"
        echo "     (Running on $CURRENT_RUN_NP physical GPUs)"
        echo "========================================================"
    fi

    # 计算本次运行需要截取的数据量
    SUBSET_SIZE=$(awk -v total="$TOTAL_GRAPHS" -v sim="$SIM_NP" -v real="$CURRENT_RUN_NP" 'BEGIN { printf("%.0f", (total / sim) * real) }')
    
    if [ "$SUBSET_SIZE" -gt "$TOTAL_GRAPHS" ]; then SUBSET_SIZE=$TOTAL_GRAPHS; fi
    if [ "$SUBSET_SIZE" -le 0 ]; then SUBSET_SIZE=100; fi
    
    echo "     Processing subset of $SUBSET_SIZE graphs (Using $CURRENT_RUN_NP processes)"

    # --- 内层循环：遍历目录下所有的查询文件 ---
    for query_path in "$QUERY_DIR"/*.dat; do
        # 检查是否真的存在文件（防止目录为空时报错）
        if [ ! -f "$query_path" ]; then continue; fi

        # 获取查询文件名 (不带扩展名)，用于创建目录
        query_filename=$(basename "$query_path")
        query_name="${query_filename%.*}"

        echo "     -> [Query] Processing: $query_filename"

        # 构建该查询图的专属输出目录
        # 结构: logs/mpi/<QueryName>/origin
        CURRENT_LOG_DIR="$BASE_OUT_DIR/$query_name"
        mkdir -p "$CURRENT_LOG_DIR/origin"
        mkdir -p "$CURRENT_LOG_DIR/ours"
        mkdir -p "$CURRENT_LOG_DIR/dynamic_ours"

        # 1. 原始 sigmo_mpi (Static)
        mpirun -np "$CURRENT_RUN_NP" \
          "$SCRIPT_DIR/build/sigmo_mpi" \
          -i 5 \
          -Q "$query_path" \
          -D "$zinc_dataset" \
          --max-data-graphs="$SUBSET_SIZE" \
          --find-all \
          >  "$CURRENT_LOG_DIR/origin/sigmo_mpi_sim${SIM_NP}_findall.log" \
          2> "$CURRENT_LOG_DIR/origin/err_sigmo_mpi_sim${SIM_NP}_findall.log"

        # 2. 你的 static ours 版本 sigmo2_mpi
        mpirun -np "$CURRENT_RUN_NP" \
          "$SCRIPT_DIR/build/sigmo2_mpi" \
          -i 5 \
          -Q "$query_path" \
          -D "$zinc_dataset" \
          --max-data-graphs="$SUBSET_SIZE" \
          --find-all \
          >  "$CURRENT_LOG_DIR/ours/sigmo_mpi_sim${SIM_NP}_findall.log" \
          2> "$CURRENT_LOG_DIR/ours/err_sigmo_mpi_sim${SIM_NP}_findall.log"   

        # 3. 你的 dynamic ours 版本 sigmo2_mpi_dynamic
        mpirun -np "$CURRENT_RUN_NP" \
          "$SCRIPT_DIR/build/sigmo2_mpi_dynamic" \
          -i 5 \
          -Q "$query_path" \
          -D "$zinc_dataset" \
          --max-data-graphs="$SUBSET_SIZE" \
          --find-all \
          >  "$CURRENT_LOG_DIR/dynamic_ours/sigmo_mpi_sim${SIM_NP}_findall.log" \
          2> "$CURRENT_LOG_DIR/dynamic_ours/err_sigmo_mpi_sim${SIM_NP}_findall.log"
    done
  done
fi
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Experiments completed in $elapsed_time seconds."
