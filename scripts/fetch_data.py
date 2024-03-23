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
import time
import pandas as pd
from dataclasses import dataclass
from datetime import datetime

if os.environ.get('CDF_LIB', '') == '':
    print('No CDF_LIB environment variable found for CDF file processing.')
    sys.exit(1)
from spacepy import pycdf


DATA_RAW_DIR = '../data_raw'
DATA_PROCESSED_DIR = '../data_processed'


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


def to_dataframe(cdf, i=-1):
    print(f"Processing CDF {i}...")

    # Helper function to preprocess data
    def prepare_data(event_type, value_key, bin_key, times):
        bins = cdf[bin_key][...]
        values = cdf[value_key][...]
        times = pd.to_datetime(times)  # Adjust unit as necessary

        # Generate records for DataFrame construction
        records = []
        for time, value_row in zip(times, values):
            for channel, value in zip(bins, value_row):
                records.append({
                    "time": time,
                    "event_type": event_type,
                    "channel": channel,
                    "value": value
                })
        return records

    # Prepare data for each event type
    electron_records = prepare_data(
        'e', 'ELECTRONS', 'ELECTRON_BINS', cdf['TIME_UTC'])
    proton_records = prepare_data(
        'p', 'PROTONS', 'PROTON_BINS', cdf['TIME_UTC'])
    dd_records = prepare_data('d', 'DD', 'DD_BINS', cdf['TIME_UTC'])

    # Combine all records into a single DataFrame
    df = pd.DataFrame(electron_records + proton_records + dd_records)
    df['channel'] = df['channel'].astype("category")
    df['event_type'] = df['event_type'].astype("category")

    return df


def pipeline():
    if 0 != subprocess.call(f"mkdir -p {DATA_RAW_DIR}",
                            shell=True):
        print("Error creating data directory", file=sys.stderr)
        return

    # Fetch data
    if 0 != subprocess.call(f"wget -r --timestamping --continue --user=ifjagh --ask-password ftp://ftptrans.psi.ch/to_radem/ -nd -np -P {DATA_RAW_DIR}",
                            shell=True):
        print("Error fetching data", file=sys.stderr)
        return

    # Create temp data directory
    temp_dir = f'../data_tmp_{str(int(time.time()))}'
    if 0 != subprocess.call(f"mkdir -p {temp_dir}",
                            shell=True):
        print(f"Error creating {temp_dir} directory", file=sys.stderr)
        return

    # Extracts all tar files
    if 0 != subprocess.call(f'for f in {DATA_RAW_DIR}/*.tar.gz; do tar -xvf "$f" -C {temp_dir}; done;',
                            shell=True):
        print("Error extracting tar files", file=sys.stderr)
        return

    # Data processing
    cdfs = [
        RawCDF(name=path.name,
               date=parse_date(path.name),
               tpe=parse_type(path.name),
               data=pycdf.CDF(str(path)))
        for path in pathlib.Path(f'{temp_dir}').rglob('*.cdf')
    ]

    science_cdfs = [cdf for cdf in cdfs if cdf.tpe == 'science']

    print(f"{len(science_cdfs)} CDFs found. Get a coffee because this may take a while...")

    df = pd.concat([to_dataframe(cdf.data, i)
                    for i, cdf in enumerate(science_cdfs)])

    # Saving data
    # Create temp data directory
    if 0 != subprocess.call(f"mkdir -p {DATA_PROCESSED_DIR}",
                            shell=True):
        print(
            f"Error creating {DATA_PROCESSED_DIR} directory", file=sys.stderr)
        return

    df.to_hdf(f'{DATA_PROCESSED_DIR}/preprocessed.h5',
              key='time', format="table")
    df.to_csv(f'{DATA_PROCESSED_DIR}/preprocessed.csv', index=False)

    # Remove tmp data files
    if 0 != subprocess.call(f'rm -rd {temp_dir}',
                            shell=True):
        print(f"Error removing {temp_dir}", file=sys.stderr)
        return

    print("DONE")


if __name__ == "__main__":
    pipeline()
