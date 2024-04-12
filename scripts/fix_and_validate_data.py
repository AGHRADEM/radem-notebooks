import pandas as pd
import pandas as pd
import numpy as np


def fix_df_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the time column to datetime and floor it to seconds, in place.
    """
    df["time"] = pd.to_datetime(df['time']).dt.floor('S')

    return df


def fix_df_time_start(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the dataframe to only include events after September 1, 2023, in place.
    """
    df.query("time >= '2023-09-01'", inplace=True)

    return df


def fix_df_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find and remove duplicates from the dataframe, in place.
    """
    df.drop_duplicates(inplace=True, keep="first")
    return df


def fix_sorting_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort the dataframe by time, in place.
    """
    df.sort_values("time", inplace=True)
    return df


def fix_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix the dataframe in place.
    """
    fix_df_time(df)
    fix_df_time_start(df)
    fix_df_duplicates(df)
    fix_sorting_df(df)
    return df


def verify_df_column_names(df: pd.DataFrame) -> None:
    """
    Verify that the dataframe has the required columns
    """

    REQUIRED_COLS = ['time', 'event_type', 'channel', 'value']

    cols = df.columns
    for col in REQUIRED_COLS:
        if col not in cols:
            raise ValueError(f"Column {col} not found in dataframe")


def verify_df_sorted(df: pd.DataFrame) -> None:
    """
    Verify that the dataframe is sorted by "time"
    """
    # Find rows where the 'time' is decreasing from the previous row
    not_sorted_mask = df['time'].diff().dt.total_seconds() < 0

    # The first row can't be "not sorted" by definition, so we can exclude it from the mask
    not_sorted_mask.iloc[0] = False

    # Filter the DataFrame to find the not sorted rows
    not_sorted_rows = df[not_sorted_mask]

    if not df['time'].is_monotonic_increasing:
        raise ValueError(
            f"Dataframe is not sorted by time:\n{not_sorted_rows}")


def verify_df_time_diffs(df: pd.DataFrame,
                         max_diff_tolerance: np.timedelta64 = np.timedelta64(
                             90, 's'),
                         min_diff_tolerance: np.timedelta64 = np.timedelta64(500, 'ms')) -> None:
    """
    Verify that the time differences between events are within tolerance.
    If time diff >= max_diff_tolerance, just prints the warning (data holes are permitted).
    If time diff <= min_diff_tolerance, raises an exception (possible floating point errors).

    Assumes that the dataframe is non-decreasingly sorted by "time".  

    There may me multiple groups of events with the same time.

    Args:
        df (pd.DataFrame): input dataframe with "time" column
        max_diff_tolerance (np.timedelta64, optional): max time difference tolerance in ms (warning only)
        min_diff_tolerance (np.timedelta64, optional): min time difference tolerance in ms (exception)

    Raises:
        ValueError: when time differences < min_diff_tolerance (possible floating point errors)
    """

    # get all unique "time" values in df
    times = df['time'].unique()

    # calc time diffs
    time_diffs = np.diff(times)

    # check if all time diffs are not larger than the tolerance
    checks = max_diff_tolerance > time_diffs
    if not all(checks):
        # find all indexes of unmet conditions
        indexes = np.where(checks == False)[0]

        # create a dataframe of times
        df_times = pd.DataFrame(times, columns=["time"])

        # find all holes
        holes = [
            f"{df_times.iloc[i]['time']} and {df_times.iloc[i + 1]['time']}" for i in indexes]

        print("Found time holes out of tolerance at times:", *holes, sep='\n\t')

    # check if all time diffs are not smaller than the tolerance
    # (possible floating point errors)
    checks = min_diff_tolerance < time_diffs
    if not all(checks):
        # find all indexes of unmet conditions
        indexes = np.where(checks == False)[0]

        # create a dataframe of times
        df_times = pd.DataFrame(times, columns=["time"])

        # find all too close values
        too_close = [
            f"{df_times.iloc[i]['time']} and {df_times.iloc[i + 1]['time']}" for i in indexes]

        raise ValueError(
            "Found time values too close to each other at times " +
            "(possible floating point errors):\n\t" +
            "\n\t".join(too_close))


def verify_df_time_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Verify that the dataframe has the correct number of events (49) for each time.
    """
    if not all(df["time"].value_counts() == 49):
        raise ValueError("Incorrect number of events for some times")
    return df


def verify_df_time_p_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Verify that the dataframe has the correct number of proton events (9) for each time.
    """
    result = df.groupby('time').apply(
        lambda group: (group['event_type'] == 'p').sum() == 9)
    if not all(result):
        raise ValueError("Incorrect number of proton events for some times")
    return df


def verify_df_time_e_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Verify that the dataframe has the correct number of electron events (9) for each time.
    """
    result = df.groupby('time').apply(
        lambda group: (group['event_type'] == 'e').sum() == 9)
    if not all(result):
        raise ValueError("Incorrect number of electron events for some times")
    return df


def verify_df_time_d_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Verify that the dataframe has the correct number of directional events (31) for each time.
    """
    result = df.groupby('time').apply(
        lambda group: (group['event_type'] == 'd').sum() == 31)
    if not all(result):
        raise ValueError(
            "Incorrect number of directional events for some times")
    return df


def verify_df(df: pd.DataFrame) -> None:
    """
    Verify the integrity of the dataframe
    """

    if df.empty:
        raise ValueError("Dataframe is empty")

    print("Verifying column names")
    verify_df_column_names(df)

    print("Verifying sorting")
    verify_df_sorted(df)

    print("Verifying time diffs")
    verify_df_time_diffs(df)

    print("Verifying time counts")
    verify_df_time_counts(df)
    print("Verifying time p counts")
    verify_df_time_d_counts(df)
    print("Verifying time e counts")
    verify_df_time_e_counts(df)
    print("Verifying time d counts")
    verify_df_time_p_counts(df)

    # df.groupby('time').apply(verify_df_time_group)
