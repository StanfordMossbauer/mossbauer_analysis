import numpy as np
from datetime import timezone

from block import get_true_time_range
from file_finder import find_files_in_range 
from mossbauer import mossbauer 
from sql import sql





def datetime_to_npdt64_utc(dt):
    dt_utc = dt.astimezone(timezone.utc)
    return np.datetime64(dt_utc.replace(tzinfo=None), "us")


def get_data_by_block_id(db, block_id,
                         directory="/data/share",
                         pattern="*.dat.*",
                         table="science_run1",
                         data_tz="America/Los_Angeles",
                         return_head=True):
    """
    Return mossbauer data in the true time range corresponding to block_id.

    Now supports one or more matched files (typically 1 or 2 files),
    and concatenates matched frames together.

    Returns
    -------
    dict | None
        {
            "block_id": int,
            "true_start": datetime,
            "true_end": datetime,
            "files": list[str],
            "file_items": [
                {"file": str, "frame_idx": np.ndarray},
                ...
            ],
            "times64": np.ndarray,
            "data": np.ndarray,
            "head": np.ndarray | None,
            "source_files": np.ndarray
        }
    """
    result = get_true_time_range(db, block_id, table=table)
    if result is None:
        return None

    t_start, t_end , mode = result
    if t_start is None or t_end is None:
        return None

    files = find_files_in_range(
        directory=directory,
        t_start=t_start,
        t_end=t_end,
        pattern=pattern,
        data_tz=data_tz
    )

    if len(files) == 0:
        return None

    t0 = datetime_to_npdt64_utc(t_start)
    t1 = datetime_to_npdt64_utc(t_end)

    all_times64 = []
    all_data = []
    all_head = [] if return_head else None
    all_source_files = []
    file_items = []

    for fname in files:
        det = mossbauer(fname)

        times64 = det.get_all_datetime64()
        mask = (times64 >= t0) & (times64 < t1)
        frame_idx = np.flatnonzero(mask)

        if frame_idx.size == 0:
            continue

        file_items.append({
            "file": fname,
            "frame_idx": frame_idx
        })

        all_times64.append(times64[mask])
        all_data.append(det.load_data()[mask])

        if return_head:
            all_head.append(det.load_head()[mask])

        all_source_files.extend([fname] * frame_idx.size)

    if len(all_times64) == 0:
        return None

    times64 = np.concatenate(all_times64, axis=0)
    data = np.concatenate(all_data, axis=0)

    if return_head:
        head = np.concatenate(all_head, axis=0)
    else:
        head = None

    source_files = np.asarray(all_source_files, dtype=object)

    # keep final result strictly ordered by time
    order = np.argsort(times64)
    times64 = times64[order]
    data = data[order]
    source_files = source_files[order]
    if return_head:
        head = head[order]

    return {
        "block_id": block_id,
        "true_start": t_start,
        "true_end": t_end,
        "files": files,
        "file_items": file_items,
        "times64": times64,
        "data": data,
        "head": head,
        "source_files": source_files,
    }