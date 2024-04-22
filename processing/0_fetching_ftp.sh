#!/usr/bin/env bash

DATA_RAW_DIR="../data_raw"

# Load environment variables
export $(grep -v '^#' ../.env | xargs)

# Fetch data from ftp
wget -r \
    --timestamping \
    --continue \
    --user=${FTP_USER} \
    --password=${FTP_PASSWORD} \
    ftp://ftptrans.psi.ch/to_radem/ \
    -nd \
    -np \
    -P ${DATA_RAW_DIR}\
    