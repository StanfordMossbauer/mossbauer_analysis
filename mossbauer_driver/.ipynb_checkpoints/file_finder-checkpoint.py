# This script is used to find files, especially data files, based on time.

import os
import re
import glob
import struct
from datetime import datetime, timedelta

HEAD40_BYTES = 40


def parse_file_datetime(fname):
    """
    Parse datetime from filenames like:
      data_20260312_134342.dat.1
      data_20260312_134342.dat.2
      data_20260312_134342.dat.3

    Rule:
      base timestamp = datetime in filename
      suffix .N means +(N-1) hours

    Returns
    -------
    datetime | None
    """
    base = os.path.basename(fname)
    m = re.match(r".*?(\d{8}_\d{6})\.dat\.(\d+)$", base)
    if not m:
        return None

    ts_str = m.group(1)
    idx = int(m.group(2))

    dt0 = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
    return dt0 + timedelta(hours=idx - 1)


def find_files_by_filename_time(directory, pattern="*.dat.*", t_start=None, t_end=None):
    """
    Find files whose filename-derived datetime falls in [t_start, t_end].

    Returns
    -------
    list[str]
    """
    paths = glob.glob(os.path.join(directory, pattern))

    parsed = []
    for p in paths:
        dt = parse_file_datetime(p)
        if dt is None:
            continue

        if t_start is not None and dt < t_start:
            continue
        if t_end is not None and dt > t_end:
            continue

        parsed.append((dt, p))

    parsed.sort(key=lambda x: x[0])
    return [p for _, p in parsed]


def read_first_frame_time(fname):
    """
    Read timestamp from the first frame only.

    File layout:
      8B SSI
      32B original header

    In original 32B header:
      word5 @ offset 20 = microseconds
      word7 @ offset 28 = seconds

    Since the file starts with:
      8B SSI + 32B header = 40B
    we read the first 40 bytes directly.

    Returns
    -------
    datetime
    """
    with open(fname, "rb") as f:
        head40 = f.read(HEAD40_BYTES)
        if len(head40) < HEAD40_BYTES:
            raise RuntimeError("file too short to contain first frame header")

    usec = struct.unpack_from("<I", head40, 28)[0]   # 8 + 20
    sec  = struct.unpack_from("<I", head40, 36)[0]   # 8 + 28

    return datetime.fromtimestamp(sec) + timedelta(microseconds=usec)


def find_files_in_range(directory, t_start, t_end, pattern="*.dat.*", expand_hours=2, verbose=False):
    """
    Find files whose FIRST FRAME timestamp falls in [t_start, t_end].

    Strategy:
      1) use filename datetime for coarse filtering
      2) use first-frame timestamp for precise filtering

    Parameters
    ----------
    directory : str
        Directory containing files.
    t_start : datetime
        Start of target range.
    t_end : datetime
        End of target range.
    pattern : str
        Glob pattern, default "*.dat.*".
    expand_hours : int
        Expand filename coarse filter window by ±expand_hours to avoid missing edge files.
    verbose : bool
        Whether to print skipped files and errors.

    Returns
    -------
    list[str]
        Matched file paths, sorted by first-frame timestamp.
    """
    if t_end < t_start:
        raise ValueError("t_end must be >= t_start")

    coarse_start = t_start - timedelta(hours=expand_hours)
    coarse_end = t_end + timedelta(hours=expand_hours)

    # Step 1: coarse filter by filename time
    candidates = find_files_by_filename_time(
        directory=directory,
        pattern=pattern,
        t_start=coarse_start,
        t_end=coarse_end,
    )

    # Step 2: precise filter by first-frame timestamp
    matched = []
    for p in candidates:
        try:
            dt = read_first_frame_time(p)
        except Exception as e:
            if verbose:
                print(f"skip {p}: {e}")
            continue

        if t_start <= dt <= t_end:
            matched.append((dt, p))

    matched.sort(key=lambda x: x[0])
    return [p for _, p in matched]