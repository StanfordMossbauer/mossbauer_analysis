# this script is used to find the related output of our sidestreams; 

# For the time being, we are reading the filter1, background, filter2, gain file
# This is useful, because our pixel 

import os
import glob
import re
import struct
from datetime import datetime, timedelta


HEAD40_BYTES = 40


def read_first_frame_time(fname):
    """
    Read timestamp from the first frame only.

    File layout:
      8B SSI
      32B original header

    In original 32B header:
      word5 @ offset 20 = microseconds
      word7 @ offset 28 = seconds
    """
    with open(fname, "rb") as f:
        head40 = f.read(HEAD40_BYTES)
        if len(head40) < HEAD40_BYTES:
            raise RuntimeError("file too short to contain first frame header")

    usec = struct.unpack_from("<I", head40, 28)[0]   # 8 + 20
    sec  = struct.unpack_from("<I", head40, 36)[0]   # 8 + 28

    return datetime.fromtimestamp(sec) + timedelta(microseconds=usec)


def parse_unix_ts_from_fname(path):
    """
    Parse the last integer before '.npy' as unix timestamp.

    Examples
    --------
    dark_2D_1773397440.npy -> 1773397440
    gain_1773397440.npy    -> 1773397440
    filter_1773397440.npy  -> 1773397440
    """
    base = os.path.basename(path)
    m = re.search(r"_(\d+)\.npy$", base)
    if not m:
        return None
    return int(m.group(1))


def find_latest_file_before_time(pattern, target_dt, verbose=False):
    """
    Find the latest file whose embedded unix timestamp is <= target_dt.

    Parameters
    ----------
    pattern : str
        Glob pattern, e.g. "/data/dark/dark_2D_*.npy"
    target_dt : datetime
        Target time.
    verbose : bool
        Print skipped files if needed.

    Returns
    -------
    str | None
        Best matched file path, or None if not found.
    """
    target_ts = target_dt.timestamp()
    paths = glob.glob(pattern)

    candidates = []
    for p in paths:
        ts = parse_unix_ts_from_fname(p)
        if ts is None:
            if verbose:
                print(f"skip {p}: cannot parse timestamp")
            continue

        if ts <= target_ts:
            candidates.append((ts, p))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def find_calib_files_for_datafile(data_fname, verbose=False):
    """
    Given a data file, use its first frame time to find matched calibration files.

    Rules:
      choose the latest file whose timestamp <= first frame time

    Targets:
      /data/dark/dark_2D_*.npy
      /data/dark/filter_8_*.npy
      /data/gain/gain_*.npy
      /data/filter/filter_*.npy

    Returns
    -------
    dict
        {
          "data_file": ...,
          "first_frame_time": datetime,
          "dark": ... or None,
          "filter8": ... or None,
          "gain": ... or None,
          "filter": ... or None,
        }
    """
    t0 = read_first_frame_time(data_fname)

    out = {
        "data_file": data_fname,
        "first_frame_time": t0,
        "dark": find_latest_file_before_time("/data/dark/dark/dark_2D_*.npy", t0, verbose=verbose),
        "filter1": find_latest_file_before_time("/data/dark/filter/filter_*.npy", t0, verbose=verbose),
        "gain": find_latest_file_before_time("/data/gain/gain/gain_*.npy", t0, verbose=verbose),
        "filter2": find_latest_file_before_time("/data/gain/filter/filter_*.npy", t0, verbose=verbose),
    }
    return out