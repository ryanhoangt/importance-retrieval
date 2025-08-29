# !/bin/bash

ROBOMIMIC_DIR="PATH_TO/square_400_paired"
OUTPUT_DIR="PATH_TO/square_400_paired_flow"

source /iliad/u/amberxie/miniconda3/bin/activate
conda activate openx

mkdir -p "$OUTPUT_DIR"

tfds build --manual_dir ${ROBOMIMIC_DIR} --data_dir ${OUTPUT_DIR}

# Now move the files to the correct location
mv ${OUTPUT_DIR}/robo_mimic/1.0.0 ${OUTPUT_DIR}
rm -r ${OUTPUT_DIR}/robo_mimic
rm -r ${OUTPUT_DIR}/downloads


echo "Processing complete."
