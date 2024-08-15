#!/usr/bin/env bash

URL=https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/
DATA_DIR="../data/kernels"

mkdir -p ${DATA_DIR}

# get data recursively, don't download existing files
wget \
    --recursive \
    --no-parent \
    --continue \
    --no-clobber \
    ${URL} \
    -P ${DATA_DIR}