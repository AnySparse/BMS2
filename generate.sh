#!/bin/bash
BASE_DIR="BMS2"
DATASET_DIR="$BASE_DIR/dataset"
QUERYSET_DIR="$BASE_DIR/queryset"
OUTPUT_BASE="$BASE_DIR/data"
SCRIPT_GEN_DATA="$BASE_DIR/generate_data.py"
SCRIPT_GEN_CS="$BASE_DIR/generate_cs.py"
if [ ! -f "$SCRIPT_GEN_DATA" ] || [ ! -f "$SCRIPT_GEN_CS" ]; then
echo "Error: Python scripts not found."
exit 1
fi
mkdir -p "$OUTPUT_BASE/data"
for file in "$DATASET_DIR"/*.smarts; do
[ -e "$file" ] || continue
filename=$(basename -- "$file")
filename_no_ext="${filename%.*}"
output_file="$OUTPUT_BASE/data/${filename_no_ext}.dat"
echo "Dataset: $filename"
python3 "$SCRIPT_GEN_DATA" -i "$file" -o "$output_file"
done
mkdir -p "$OUTPUT_BASE/query"
mkdir -p "$OUTPUT_BASE/cs"
for file in "$QUERYSET_DIR"/*.smarts; do
[ -e "$file" ] || continue
filename=$(basename -- "$file")
filename_no_ext="${filename%.*}"
echo "Queryset: $filename"
output_data_file="$OUTPUT_BASE/query/${filename_no_ext}.dat"
python3 "$SCRIPT_GEN_DATA" -i "$file" -o "$output_data_file"
output_cs_dir="$OUTPUT_BASE/cs/${filename_no_ext}"
if [[ "$filename_no_ext" == "data" ]] || [[ "$filename_no_ext" == "query" ]] || [[ "$filename_no_ext" == "cs" ]]; then
output_cs_dir="${output_cs_dir}_cs_out"
fi
python3 "$SCRIPT_GEN_CS" -i "$file" -o "$output_cs_dir"
done
echo "All tasks finished."
