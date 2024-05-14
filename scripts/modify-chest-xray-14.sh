#!/bin/bash

# Define the base directory and the target directory
BASE_DIR="/cluster/home/taheeraa/datasets/chestxray-14"
TARGET_DIR="${BASE_DIR}/images"

# Create the target directory if it does not exist
mkdir -p "${TARGET_DIR}"

# Loop through all subdirectories in the format 'imagesXYZ/images'
for SUBDIR in ${BASE_DIR}/images*/images; do
    echo "Moving files from ${SUBDIR} to ${TARGET_DIR}"
    # Move all files from each subdirectory to the target directory
    mv "${SUBDIR}"/* "${TARGET_DIR}/"
done

echo "All files have been moved to ${TARGET_DIR}"
