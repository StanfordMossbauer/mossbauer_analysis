from datetime import datetime, timezone, timedelta
from sql import get_values_in_timerange
import numpy as np

# Give a base time, generate the corresponding block id
def get_block_index(target_datetime):
    """
    Take caution that 
    Return:
        integer number of 5-minute intervals between target_datetime (converted to UTC)
        and 2026-01-01 00:00:00 Los Angeles Time 
    """
    time_utc = target_datetime.astimezone(timezone.utc)
    ref_utc = datetime(2026, 1, 1, 8, 0, 0, tzinfo=timezone.utc)

    delta = time_utc - ref_utc
    return int(delta.total_seconds() // 300)


# Given a block id, return the corresponding UTC time
def get_time_utc_from_block_id(block_id):
    """
    Return:
        UTC datetime corresponding to the given block_id,
        where each block is 5 minutes from 2026-01-01 00:00:00 Los Angeles; 
    """
    ref_utc = datetime(2026, 1, 1, 8, 0, 0, tzinfo=timezone.utc)
    return ref_utc + timedelta(minutes=5 * block_id)

def get_local_time_from_block_id(block_id):    
    return get_time_utc_from_block_id(block_id).astimezone()


def get_true_time_range(db, block_id, table="science_run1",
                        lookback_minutes=2,
                        lookahead_minutes=20):
    coarse_start = get_time_utc_from_block_id(block_id-1)
    coarse_end = get_time_utc_from_block_id(block_id )

    times, data = get_values_in_timerange(
        db,
        t_start=coarse_start - timedelta(minutes=lookback_minutes),
        t_end=coarse_end + timedelta(minutes=lookahead_minutes),
        value_cols=("sp_current_set", "Vpp_set"),
        table=table
    )

    if len(times) < 2:
        return None

    sp = np.asarray(data["sp_current_set"])
    vpp = np.asarray(data["Vpp_set"])
    t = np.asarray(times, dtype=object)

    change_mask = (sp[1:] != sp[:-1]) | (vpp[1:] != vpp[:-1])
    change_idx = np.flatnonzero(change_mask) + 1

    if change_idx.size == 0:
        return None

    in_block = (t[change_idx] >= coarse_start) & (t[change_idx] < coarse_end)
    start_candidates = change_idx[in_block]

    if start_candidates.size == 0:
        return None

    start_idx = start_candidates[0]
    later_changes = change_idx[change_idx > start_idx]

    true_start = t[start_idx]
    true_end = t[later_changes[0]] if later_changes.size > 0 else None

    # We accept an error of 4s here, because it is 300s or 500s (old)
    if true_end is not None:
        duration_s = (true_end - true_start).total_seconds()
        nearest_100 = int(duration_s / 100.0 + 0.5) * 100
        if abs(duration_s - nearest_100) <= 4:
            true_end = true_start + timedelta(seconds=nearest_100)

    sp_prev = sp[start_idx - 1]
    sp_now = sp[start_idx]
    vpp_prev = vpp[start_idx - 1]
    vpp_now = vpp[start_idx]

    if sp_prev <= 0 and sp_now > 0:
        block_type = +1
    elif sp_now < 0:
        block_type = -1
    elif sp_now == 0 and vpp_now != vpp_prev:
        block_type = 0
    else:
        block_type = None

    return true_start, true_end , block_type


