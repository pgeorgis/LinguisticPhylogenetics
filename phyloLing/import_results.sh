#!/bin/bash

# Ensure correct usage
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 DATASET_NAME EXP_DIR"
    exit 1
fi

DATASET_NAME=$1
EXP_DIR=$2
DATASET_DIR="datasets/$DATASET_NAME"

# Remove existing phone_corr directory
rm -r "$DATASET_DIR/phone_corr"

# Copy phone_corr from EXP_DIR to DATASET_DIR
cp -r "$EXP_DIR/phone_corr" "$DATASET_DIR"

# Ensure target directories exist
mkdir -p "$DATASET_DIR/trees"
mkdir -p "$DATASET_DIR/dist-matrix"

# Copy files to respective directories
cp "$EXP_DIR/tree.png" "$DATASET_DIR/trees"
cp "$EXP_DIR/newick.tre" "$DATASET_DIR/trees"
cp "$EXP_DIR/distance-matrix.tsv" "$DATASET_DIR/dist-matrix"

echo "Successfully imported experimental results from $EXP_DIR"
