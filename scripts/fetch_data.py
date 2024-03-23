#!/usr/bin/env python3
"""
This script is responsible for fetching data from the PSI FTP server
and saving it to a HDF5 and CSV file.

File should be run from /scripts directory.
"""

import subprocess
import sys
import os
import pathlib
import pandas as pd
from dataclasses import dataclass
from datetime import datetime

if os.environ.get('CDF_LIB', '') == '':
    print('No CDF_LIB environment variable found for CDF file processing.')
    sys.exit(1)
from spacepy import pycdf


@dataclass
class RawCDF:
    name: str
    date: datetime
    tpe: str  # type
    data: pycdf.CDF

    def count_events(self) -> int:
        total_channels = 31 + 9 + 9  # TODO: Rewrite in terms of cdf Vars
        return total_channels * len(self.data["TIME_UTC"])


def parse_date(filename: str) -> datetime:
    date_string = filename[-12:-4]
    format = '%Y%m%d'
    return datetime.strptime(date_string, format).date()


def parse_type(filename: str) -> str:
    # FixMe: Non exhaustive match
    return 'science' if filename[8:10] == 'sc' else 'housekeeping'


def to_dataframe(cdf: pycdf.CDF) -> pd.DataFrame:
    # Read electron channels
    electron_df = pd.concat([
        pd.DataFrame({
            "time": pd.to_datetime(str(time)),
            "event_type": 'e',
            "channel": list(cdf["ELECTRON_BINS"]),
            "value": electrons
        }) for electrons, time in zip(cdf["ELECTRONS"], cdf["TIME_UTC"])
    ])

    # Read proton channels
    proton_df = pd.concat([
        pd.DataFrame({
            "time": pd.to_datetime(str(time)),
            "event_type": 'p',
            "channel": list(cdf["PROTON_BINS"]),
            "value": protons
        }) for protons, time in zip(cdf["PROTONS"], cdf["TIME_UTC"])
    ])

    # Read DD channels
    dd_df = pd.concat([
        pd.DataFrame({
            "time": pd.to_datetime(str(time)),
            "event_type": 'd',
            "channel": list(cdf["DD_BINS"]),
            "value": dd
        }) for dd, time in zip(cdf["DD"], cdf["TIME_UTC"])
    ])

    df = pd.concat([electron_df, proton_df, dd_df])
    df['channel'] = df['channel'].astype("category")
    df['event_type'] = df['event_type'].astype("category")

    return df


def pipeline():
    if 0 != subprocess.call("mkdir -p ../data",
                            shell=True):
        print("Error creating data directory", file=sys.stderr)
        return

    # Fetch data
    if 0 != subprocess.call("wget -r --user=ifjagh --ask-password ftp://ftptrans.psi.ch/to_radem/ -nd -np -P ../data/",
                            shell=True):
        print("Error fetching data", file=sys.stderr)
        return

    # Extracts all tar files from data/ directory
    if 0 != subprocess.call('for f in ../data/*.tar.gz; do tar -xvf "$f" -C ../data/; done;',
                            shell=True):
        print("Error extracting tar files", file=sys.stderr)
        return

    # Remove tar.gz files and all non-raw data
    if 0 != subprocess.call('find ../data -maxdepth 1 -type f -delete',
                            shell=True):
        print("Error removing tar files", file=sys.stderr)
        return

    cdfs = [
        RawCDF(name=path.name,
               date=parse_date(path.name),
               tpe=parse_type(path.name),
               data=pycdf.CDF(str(path)))
        for path in pathlib.Path('../data').rglob('*.cdf')
    ]

    science_cdfs = [cdf for cdf in cdfs if cdf.tpe == 'science']

    df = pd.concat([to_dataframe(cdf.data)
                    for _, cdf in enumerate(science_cdfs)])

    df.to_hdf('../data/preprocessed.h5', key='time', format="table")
    df.to_csv('../data/preprocessed.csv', index=False)

    df = pd.read_hdf('../data/preprocessed.h5')
    print(df)
    print("DONE")


if __name__ == "__main__":
    pipeline()
