#!/usr/bin/env bash

echo "============= FETCHING DATA =============" 
./fetch_all.sh

echo "============= RADEM - PROCESSING & UPLOADING =============" 
cd scripts
python3 ./1_processing_raw.py
python3 ./2_processing_csv.py
python3 ./3_processing_line_protocol.py
cd ..

echo ok