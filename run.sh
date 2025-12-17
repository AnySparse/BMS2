#!/bin/bash


BASE_DIR="BMS2"


EXEC_CMD="$BASE_DIR/build/BMS2"


DATA_DIR="$BASE_DIR/data/data"
QUERY_DIR="$BASE_DIR/data/query"
CS_DIR="$BASE_DIR/data/cs"

LOG_BASE="$BASE_DIR/logs"


COMMON_ARGS="-i 6 -p -c query --find-all --max-data-graphs 1000000000"

if [ ! -f "$EXEC_CMD" ]; then
    echo "Error: Executable not found at $EXEC_CMD"
    exit 1
fi

echo "Starting Batch Processing..."
echo "Logs will be organized by DATASET in: $LOG_BASE"


for data_path in "$DATA_DIR"/*.dat; do
    [ -e "$data_path" ] || continue
    data_name=$(basename "$data_path" .dat)

    echo "=================================================="
    echo "Processing Dataset: $data_name"
    echo "=================================================="

    
    CURRENT_DATA_LOG_ROOT="$LOG_BASE/$data_name"
    mkdir -p "$CURRENT_DATA_LOG_ROOT"

    
    echo "  [Phase 1] Standard Queries..."
    

    PHASE1_LOG_DIR="$CURRENT_DATA_LOG_ROOT/standard"
    mkdir -p "$PHASE1_LOG_DIR"


    for query_path in "$QUERY_DIR"/*.dat; do
        [ -e "$query_path" ] || continue
        query_name=$(basename "$query_path" .dat)


        log_out="$PHASE1_LOG_DIR/${query_name}.log"
        log_err="$PHASE1_LOG_DIR/${query_name}.err"


        $EXEC_CMD $COMMON_ARGS \
            -D "$data_path" \
            -Q "$query_path" \
            > "$log_out" 2> "$log_err"
    done

    
    echo "  [Phase 2] CS Queries..."

    
    for cs_subdir in "$CS_DIR"/*; do
        [ -d "$cs_subdir" ] || continue
        subdir_name=$(basename "$cs_subdir")

        
        PHASE2_LOG_DIR="$CURRENT_DATA_LOG_ROOT/cs/$subdir_name"
        mkdir -p "$PHASE2_LOG_DIR"

        for query_path in "$cs_subdir"/*.dat; do
            [ -e "$query_path" ] || continue
            query_filename=$(basename "$query_path")       
            query_name_noext=$(basename "$query_path" .dat) 

            
            log_out="$PHASE2_LOG_DIR/${query_name_noext}.log"
            log_err="$PHASE2_LOG_DIR/${query_name_noext}.err"

            
            if [ "$query_filename" == "cluster_other.dat" ]; then
                  $EXEC_CMD $COMMON_ARGS \
                    -D "$data_path" \
                    -Q "$query_path" \
                    > "$log_out" 2> "$log_err"
            else
                
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
