#!/usr/bin/env bash

URL=http://srem.psi.ch/datarepo/V0/irem/
DATA_DIR="../data_srem"

mkdir -p ${DATA_DIR}

wget \
    --recursive \
    --no-parent \
    --continue \
    --no-clobber \
    -A gz \
    ${URL} \
    -P ${DATA_DIR}

rm -rd ../data_srem/srem.psi.ch/datarepo/V0/irem/summaryplots