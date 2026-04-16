# This script is used to find files, especially data files, based on time.

import os
import re
import glob
import struct
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

HEAD40_BYTES = 40
UTC= timezone.utc


# We are now using the UTC time for anything, not the local time; 
# However, the time is not so precise, but it is still useful to have some basic judgements; 
def parse_file_datetime(fname, data_tz="America/Los_Angeles", L0_bug_fixed=False):
    """
    The name of the filename is the time that we start the acquisition, and each hour the appendix of it increases by one; 
    Take caution that the L0 processed data does not apply this rule, because the file length of it is set incorrectly, and I do not want to restart the script; 
    """
    base = os.path.basename(fname)
    m = re.match(r".*?(\d{8}_\d{6})\.dat\.(\d+)$", base)
    if not m:
        return None

    # The ts_str is the initial time, and the idx is the time offset of it. 
    ts_str = m.group(1)
    idx = int(m.group(2))

    # Parse as naive local time
    dt_local = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
    dt_local = dt_local.replace(tzinfo=ZoneInfo(data_tz))

    if base.upper().startswith("L0") and not L0_bug_fixed:
        hour_per_index = 270376 / 274996
        dt_local = dt_local + timedelta(hours=(idx - 1) * hour_per_index)
    else:
        dt_local = dt_local + timedelta(hours=idx - 1)

    # Convert to UTC
    return dt_local.astimezone(timezone.utc)


# This is the basic function to find the file based on the filename; 
def find_files_by_filename_time(directory='/data/share', pattern="*.dat.*", t_start=None, t_end=None, data_tz="America/Los_Angeles"):
    """
    Find files whose filename-derived datetime falls in [t_start, t_end].
    This includes the files that could include the data in the t_start and t_end range; 
    Returns the files list; 
    """
    paths = glob.glob(os.path.join(directory, pattern))

    # We always use the UTC timezone here; 
    if t_start is not None:
        t_start = t_start.astimezone(UTC)
    if t_end is not None:
        t_end = t_end.astimezone(UTC)

    parsed = []
    for p in paths:
        dt = parse_file_datetime(p,data_tz= data_tz)
        if dt is None:
            continue

        if t_start is not None and dt < t_start:
            continue
        if t_end is not None and dt > t_end:
            continue
        parsed.append((dt, p))

    parsed.sort(key=lambda x: x[0])
    return [p for _, p in parsed]




# A more precise way;
def read_first_frame_time(fname):
    """
    word5 @ offset 28 = microseconds
    word7 @ offset 36 = seconds
    Returns the utc timestamp; 
    """
    with open(fname, "rb") as f:
        head40 = f.read(HEAD40_BYTES)
        if len(head40) < HEAD40_BYTES:
            raise RuntimeError("file too short to contain first frame header")

    usec = struct.unpack_from("<I", head40, 28)[0]   # 8 + 20
    sec  = struct.unpack_from("<I", head40, 36)[0]   # 8 + 28

    return datetime.fromtimestamp(sec,tz=timezone.utc) + timedelta(microseconds=usec)


def find_files_in_range(directory='/data/share', t_start= None, t_end= None, pattern="*.dat.*", expand_hours=10, verbose=False,data_tz="America/Los_Angeles"):
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


    if t_start is None:
        t_start = datetime.now() - timedelta(hours=2)
    if t_end is None:
        t_end = datetime.now() - timedelta(hours=1)

    if t_end < t_start:
        raise ValueError("t_end must be >= t_start")

    # Still, we use UTC time here. 
    if t_start is not None:
        t_start = t_start.astimezone(UTC)
    if t_end is not None:
        t_end = t_end.astimezone(UTC)



    coarse_start = t_start - timedelta(hours=expand_hours)
    coarse_end = t_end + timedelta(hours=expand_hours)

    # Step 1: coarse filter by filename time
    candidates = find_files_by_filename_time(
        directory=directory,
        pattern=pattern,
        t_start=coarse_start,
        t_end=coarse_end,
        data_tz=data_tz, 
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

        # If the first-frame timestamp is within one hour, it is still possible that it includes some data from it; 
        if t_start - timedelta(hours=1) <= dt < t_end:
            matched.append((dt, p))

    matched.sort(key=lambda x: x[0])
    return [p for _, p in matched]