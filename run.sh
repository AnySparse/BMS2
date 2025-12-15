#!/bin/bash

# ================= 基础配置 =================
# 项目根目录
BASE_DIR="/home/hujiayue/BMS2"

# 可执行程序路径
EXEC_CMD="$BASE_DIR/build/BMS2"

# 输入数据路径
DATA_DIR="$BASE_DIR/data/data"
QUERY_DIR="$BASE_DIR/data/query"
CS_DIR="$BASE_DIR/data/cs"

# 日志输出根目录
LOG_BASE="$BASE_DIR/logs"

# 固定通用参数
COMMON_ARGS="-i 6 -p -c query --find-all --max-data-graphs 1000000000"

# ================= 环境检查 =================
if [ ! -f "$EXEC_CMD" ]; then
    echo "Error: Executable not found at $EXEC_CMD"
    exit 1
fi

echo "Starting Batch Processing..."
echo "Logs will be organized by DATASET in: $LOG_BASE"

# ================= 开始处理 =================

# 最外层循环：遍历每一个数据集 (.dat)
for data_path in "$DATA_DIR"/*.dat; do
    [ -e "$data_path" ] || continue
    data_name=$(basename "$data_path" .dat)

    echo "=================================================="
    echo "Processing Dataset: $data_name"
    echo "=================================================="

    # 为当前数据集创建根日志目录: logs/数据集名/
    CURRENT_DATA_LOG_ROOT="$LOG_BASE/$data_name"
    mkdir -p "$CURRENT_DATA_LOG_ROOT"

    # --------------------------------------------------
    # 阶段 1: 运行 Standard Query (无 use-cs)
    # --------------------------------------------------
    echo "  [Phase 1] Standard Queries..."
    
    # 创建 standard 目录: logs/数据集名/standard/
    PHASE1_LOG_DIR="$CURRENT_DATA_LOG_ROOT/standard"
    mkdir -p "$PHASE1_LOG_DIR"

    # 遍历每一个查询文件
    for query_path in "$QUERY_DIR"/*.dat; do
        [ -e "$query_path" ] || continue
        query_name=$(basename "$query_path" .dat)

        # 日志文件: logs/数据集名/standard/查询名.log
        log_out="$PHASE1_LOG_DIR/${query_name}.log"
        log_err="$PHASE1_LOG_DIR/${query_name}.err"

        # 执行命令
        $EXEC_CMD $COMMON_ARGS \
            -D "$data_path" \
            -Q "$query_path" \
            > "$log_out" 2> "$log_err"
    done

    # --------------------------------------------------
    # 阶段 2: 运行 CS Query (部分 use-cs)
    # --------------------------------------------------
    echo "  [Phase 2] CS Queries..."

    # 遍历 cs 下的所有子文件夹 (例如: query1_cs_out)
    for cs_subdir in "$CS_DIR"/*; do
        [ -d "$cs_subdir" ] || continue
        subdir_name=$(basename "$cs_subdir")

        # 创建 CS 子目录: logs/数据集名/cs/子文件夹名/
        PHASE2_LOG_DIR="$CURRENT_DATA_LOG_ROOT/cs/$subdir_name"
        mkdir -p "$PHASE2_LOG_DIR"

        # 遍历子文件夹内的所有 .dat
        for query_path in "$cs_subdir"/*.dat; do
            [ -e "$query_path" ] || continue
            query_filename=$(basename "$query_path")       # 带后缀 (cluster_other.dat)
            query_name_noext=$(basename "$query_path" .dat) # 不带后缀 (cluster_other)

            # 日志文件: logs/数据集名/cs/子文件夹名/文件名.log
            log_out="$PHASE2_LOG_DIR/${query_name_noext}.log"
            log_err="$PHASE2_LOG_DIR/${query_name_noext}.err"

            # 逻辑判断
            if [ "$query_filename" == "cluster_other.dat" ]; then
                # 特殊情况：cluster_other.dat -> 不带 --use-cs
                $EXEC_CMD $COMMON_ARGS \
                    -D "$data_path" \
                    -Q "$query_path" \
                    > "$log_out" 2> "$log_err"
            else
                # 正常情况 -> 带 --use-cs
                $EXEC_CMD $COMMON_ARGS \
                    -D "$data_path" \
                    -Q "$query_path" \
                    --use-cs \
                    > "$log_out" 2> "$log_err"
            fi
        done
    done

    echo "  -> Finished dataset: $data_name"
    echo ""
done

echo "All datasets processed."