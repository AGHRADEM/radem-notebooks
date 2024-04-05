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
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import argparse

if os.environ.get('CDF_LIB', '') == '':
    print('No CDF_LIB environment variable found for CDF file processing.')
    sys.exit(1)
from spacepy import pycdf


DATA_RAW_DIR = pathlib.Path('../data_raw')
DATA_PROCESSED_DIR = pathlib.Path('../data_processed')
OUTPUT_FILE_WITHOUT_EXT = pathlib.Path('preprocessed')


@dataclass
class RawCDF:
    name: str
    date: datetime
    type_: str
    data: pycdf.CDF

    def count_events(self) -> int:
        # TODO: Rewrite in terms of cdf Vars
        total_channels = 31 + 9 + 9
        return total_channels * len(self.data["TIME_UTC"])

    @staticmethod
    def parse_date(filename: pathlib.Path) -> datetime.date:
        date_string = filename[-12:-4]
        format_ = '%Y%m%d'
        dt = datetime.strptime(date_string, format_).date()
        return dt

    @staticmethod
    def parse_type(filename: pathlib.Path) -> str:
        # FixMe: Non exhaustive match
        return 'science' if filename[8:10] == 'sc' else 'housekeeping'

    def to_dataframe(self, i=-1):
        print(f"Processing CDF {i}...")
        cdf = self.data

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


def make_dir(dir: pathlib.Path) -> bool:
    if 0 != subprocess.call(f"mkdir -p {dir}",
                            shell=True):
        print(f"Error creating {dir} directory", file=sys.stderr)
        return False
    return True


def remove_dir(dir: pathlib.Path) -> bool:
    if 0 != subprocess.call(f"rm -rd {dir}",
                            shell=True):
        print(f"Error removing {dir} directory", file=sys.stderr)
        return False
    return True


def remove_file(file: pathlib.Path) -> bool:
    if 0 != subprocess.call(f"rm -f {file}",
                            shell=True):
        print(f"Error removing {file} file", file=sys.stderr)
        return False
    return True


def fetch_data(target_dir: pathlib.Path) -> bool:
    if 0 != subprocess.call(f"wget -r --timestamping --continue --user=ifjagh --ask-password ftp://ftptrans.psi.ch/to_radem/ -nd -np -P {target_dir}",
                            shell=True):
        print("Error fetching data", file=sys.stderr)
        return False
    return True


def extract_data(source_dir: pathlib.Path, target_dir: pathlib.Path) -> bool:
    if 0 != subprocess.call(f"for f in {source_dir}/*.tar.gz; do tar -xvf \"$f\" -C {target_dir}; done;",
                            shell=True):
        print("Error extracting tar files", file=sys.stderr)
        return False
    return True


def process_data(source_dir: pathlib.Path) -> pd.DataFrame:
    cdfs = [
        RawCDF(name=path.name,
               date=RawCDF.parse_date(path.name),
               type_=RawCDF.parse_type(path.name),
               data=pycdf.CDF(str(path)))
        for path in pathlib.Path(f'{source_dir}').rglob('*.cdf')
    ]

    science_cdfs = [cdf for cdf in cdfs if cdf.type_ == 'science']
    science_cdfs.sort(key=lambda cdf: cdf.date)
    print(science_cdfs)

    print(f"{len(science_cdfs)} CDFs found. Get a coffee because this may take a while...")

    A = [cdf.to_dataframe(i)
         for i, cdf in enumerate(science_cdfs)]

    df = pd.concat(A)
    df.sort_values(by=['time', 'event_type', 'channel'], inplace=True)
    return df


def filter_data(df: pd.DataFrame) -> None:
    df.query("time >= '2023-09-01'", inplace=True)


def save_csv_data(df: pd.DataFrame, target_dir: pathlib.Path, file_without_ext: str):
    df.to_csv(f'{target_dir}/{file_without_ext}.csv', index=False)


def save_hdf_data(df: pd.DataFrame, target_dir: pathlib.Path, file_without_ext: str):
    df.to_hdf(f'{target_dir}/{file_without_ext}.h5',
              key='time', format="table")


def args_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Fetch data from PSI FTP server and save it to HDF5 and CSV files.')
    parser.add_argument('--skip-fetch', action='store_true',
                        help='Skip fetching data from PSI FTP server')
    parser.add_argument('--skip-cleanup', action='store_true',
                        help='Skip cleanup of temporary files')
    return parser.parse_args()


def pipeline(args: argparse.Namespace | None = None):
    if args is None:
        args = args_parser()

    try:
        # "../data_tmp_1711394609"
        temp_dir = pathlib.Path(f'../data_tmp_{int(time.time())}')
        if not make_dir(temp_dir):
            return

        if not make_dir(DATA_RAW_DIR):
            return

        if not args.skip_fetch:
            if not fetch_data(DATA_RAW_DIR):
                return

        if not extract_data(DATA_RAW_DIR, temp_dir):
            return

        df = process_data(temp_dir)

        filter_data(df)

        if not make_dir(DATA_PROCESSED_DIR):
            return

        save_csv_data(df, DATA_PROCESSED_DIR, OUTPUT_FILE_WITHOUT_EXT)
        save_hdf_data(df, DATA_PROCESSED_DIR, OUTPUT_FILE_WITHOUT_EXT)

        print("SUCCESS")
    except KeyboardInterrupt:
        print("Interrupted by user")
        remove_file(f'{DATA_PROCESSED_DIR}/{OUTPUT_FILE_WITHOUT_EXT}.csv')
        remove_file(f'{DATA_PROCESSED_DIR}/{OUTPUT_FILE_WITHOUT_EXT}.h5')
    finally:
        if not args.skip_cleanup:
            if not remove_dir(temp_dir):
                return


if __name__ == "__main__":
    args = args_parser()
    pipeline(args)
