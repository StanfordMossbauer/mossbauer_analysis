from datetime import datetime, timezone, timedelta
from sql import get_values_in_timerange
import numpy as np
from sql import sql

from file_finder import find_files_in_range

from L3 import L3Driver
from L0 import L0Driver
from L2P import L2ParaDriver
from S2 import L2SpectrumDriver
from raw import epix


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
    # Convert the time_utc into the local time   
    return get_time_utc_from_block_id(block_id).astimezone()


def get_true_time_range(db, block_id, table="science_run1",
                        lookback_minutes=2,
                        lookahead_minutes=20):
    # This function finds the real time range ;
    # Update: find the block that starts within this block; 
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


# Maybe I could re-organize the functions about the blocks here into one file ; 
# The Block utils, single block reader, double block reader together 

class BlockDataReader:
    def __init__(self,
                 db=None,
                 driver_cls=L3Driver,
                 directory="/data/share",
                 pattern="*.dat.*",
                 table="science_run1",
                 data_tz="America/Los_Angeles"):

        self.db = sql() if db is None else db
        self.driver_cls = driver_cls
        self.directory = directory
        self.pattern = pattern
        self.table = table
        self.data_tz = data_tz

        self.clear()

    def clear(self):
        self.block_id = None
        self.true_start = None
        self.true_end = None
        self.block_type = None

        self.files = []
        self.file_items = []

        self.times64 = None
        self.data = None
        self.head = None

        self.used_driver_cls = None
        self.used_directory = None
        self.used_pattern = None
        self.used_table = None
        self.used_data_tz = None

    @property
    def success(self):
        return (self.data is not None) and (self.head is not None)

    @staticmethod
    def datetime_to_npdt64_utc(dt):
        dt_utc = dt.astimezone(timezone.utc)
        return np.datetime64(dt_utc.replace(tzinfo=None), "us")

    def get_true_time_range(self, block_id, table=None):
        table = self.table if table is None else table
        return get_true_time_range(self.db, block_id, table=table)

    def find_files(self, t_start, t_end, directory=None, pattern=None, data_tz=None):
        directory = self.directory if directory is None else directory
        pattern = self.pattern if pattern is None else pattern
        data_tz = self.data_tz if data_tz is None else data_tz

        return find_files_in_range(
            directory=directory,
            t_start=t_start,
            t_end=t_end,
            pattern=pattern,
            data_tz=data_tz
        )

    def fetch_block(self, block_id,
                    directory=None,
                    driver_cls=None,
                    pattern=None,
                    table=None,
                    data_tz=None):
        self.clear()
        self.block_id = block_id

        driver_cls = self.driver_cls if driver_cls is None else driver_cls
        directory = self.directory if directory is None else directory
        pattern = self.pattern if pattern is None else pattern
        table = self.table if table is None else table
        data_tz = self.data_tz if data_tz is None else data_tz

        self.used_driver_cls = driver_cls
        self.used_directory = directory
        self.used_pattern = pattern
        self.used_table = table
        self.used_data_tz = data_tz

        result = self.get_true_time_range(block_id, table=table)
        if result is None:
            return

        t_start, t_end, block_type = result
        if t_start is None or t_end is None:
            return

        self.true_start = t_start
        self.true_end = t_end
        self.block_type = block_type

        files = self.find_files(
            t_start,
            t_end,
            directory=directory,
            pattern=pattern,
            data_tz=data_tz
        )
        if len(files) == 0:
            return

        self.files = files

        t0 = self.datetime_to_npdt64_utc(t_start)
        t1 = self.datetime_to_npdt64_utc(t_end)

        file_items = []
        total_n = 0

        for fname in files:
            det = driver_cls(fname)
            times64 = det.get_all_datetime64()

            left = np.searchsorted(times64, t0, side="left")
            right = np.searchsorted(times64, t1, side="left")

            if left >= right:
                continue

            n = int(right - left)
            file_items.append({
                "file": fname,
                "start": int(left),
                "stop": int(right),
                "n": n,
            })
            total_n += n

        if total_n == 0:
            return

        self.file_items = file_items

        data_out = None
        head_out = None
        pos = 0

        for item in file_items:
            det = driver_cls(item["file"])

            data_chunk = det.load_data()[item["start"]:item["stop"]]
            head_chunk = det.load_head()[item["start"]:item["stop"]]
            n = item["n"]

            if data_out is None:
                data_shape = (total_n,) + data_chunk.shape[1:]
                data_out = np.empty(data_shape, dtype=data_chunk.dtype)

            if head_out is None:
                head_shape = (total_n,) + head_chunk.shape[1:]
                head_out = np.empty(head_shape, dtype=head_chunk.dtype)

            data_out[pos:pos+n] = data_chunk
            head_out[pos:pos+n] = head_chunk
            pos += n

        self.data = data_out
        self.head = head_out

class DoubleBlockDataReader:
    def __init__(self,
                 db=None,
                 reader_cls=BlockDataReader,
                 pattern="*.dat.*",
                 table="science_run1",
                 data_tz="America/Los_Angeles"):
        
        self.db = sql() if db is None else db
        self.reader_cls = reader_cls

        self.driver_cls = L3Driver
        self.directory = "/data/share"
        self.pattern = pattern
        self.table = table
        self.data_tz = data_tz

        self.clear()

    def clear(self):
        self.anchor_block_id = None
        self.anchor_block_type = None

        self.mode = None
        self.block_ids = []

        self.forward_block_id = None
        self.backward_block_id = None

        self.single_reader = None
        self.forward_reader = None
        self.backward_reader = None
    
    @property
    def success(self):
        if self.mode == "single":
            return self.single_reader is not None and self.single_reader.success
        if self.mode == "double":
            return (
                self.forward_reader is not None and self.forward_reader.success and
                self.backward_reader is not None and self.backward_reader.success
            )
        return False

    def _make_reader(self):
        return self.reader_cls(
            db=self.db,
            driver_cls=self.driver_cls,
            directory=self.directory,
            pattern=self.pattern,
            table=self.table,
            data_tz=self.data_tz,
        )

    def _get_block_meta(self, block_id):
        reader = self._make_reader()
        result = reader.get_true_time_range(block_id, table=self.table)
        if result is None:
            return None

        t_start, t_end, block_type = result
        if t_start is None or t_end is None:
            return None

        return {
            "block_id": block_id,
            "true_start": t_start,
            "true_end": t_end,
            "block_type": block_type,
        }

    def _fetch_one(self, block_id):
        reader = self._make_reader()
        reader.fetch_block(
            block_id,
            directory=self.directory,
            driver_cls=self.driver_cls,
            pattern=self.pattern,
            table=self.table,
            data_tz=self.data_tz,
        )
        return reader

    def fetch_block(self, anchor_block_id):
        self.clear()
        self.anchor_block_id = anchor_block_id

        meta = self._get_block_meta(anchor_block_id)
        if meta is None:
            return

        self.anchor_block_type = meta["block_type"]

        # Standalone block
        if meta["block_type"] == 0:
            reader = self._fetch_one(anchor_block_id)
            if reader.success:
                self.mode = "single"
                self.block_ids = [anchor_block_id]
                self.single_reader = reader
            return

        # Forward block, next one must be backward
        if meta["block_type"] == 1:
            forward_block_id = anchor_block_id
            backward_block_id = anchor_block_id + 1

            backward_meta = self._get_block_meta(backward_block_id)
            if backward_meta is None or backward_meta["block_type"] != -1:
                raise ValueError(
                    f"Block {anchor_block_id} has type +1, but block {backward_block_id} is not -1."
                )

        # Backward block, previous one must be forward
        elif meta["block_type"] == -1:
            forward_block_id = anchor_block_id - 1
            backward_block_id = anchor_block_id

            forward_meta = self._get_block_meta(forward_block_id)
            if forward_meta is None or forward_meta["block_type"] != 1:
                raise ValueError(
                    f"Block {anchor_block_id} has type -1, but block {forward_block_id} is not +1."
                )

        else:
            raise ValueError(
                f"Unexpected block_type={meta['block_type']} for block {anchor_block_id}."
            )

        forward_reader = self._fetch_one(forward_block_id)
        backward_reader = self._fetch_one(backward_block_id)

        if not forward_reader.success or not backward_reader.success:
            return

        if forward_reader.block_type != 1:
            raise ValueError(
                f"Block {forward_block_id} is expected to be +1, but got {forward_reader.block_type}."
            )

        if backward_reader.block_type != -1:
            raise ValueError(
                f"Block {backward_block_id} is expected to be -1, but got {backward_reader.block_type}."
            )

        self.mode = "double"
        self.block_ids = [forward_block_id, backward_block_id]

        self.forward_block_id = forward_block_id
        self.backward_block_id = backward_block_id

        self.forward_reader = forward_reader
        self.backward_reader = backward_reader

    @property
    def data(self):
        if self.mode == "single" and self.single_reader is not None:
            return self.single_reader.data
        return None

    @property
    def head(self):
        if self.mode == "single" and self.single_reader is not None:
            return self.single_reader.head
        return None

    @property
    def forward_data(self):
        if self.forward_reader is None:
            return None
        return self.forward_reader.data

    @property
    def backward_data(self):
        if self.backward_reader is None:
            return None
        return self.backward_reader.data

    @property
    def forward_head(self):
        if self.forward_reader is None:
            return None
        return self.forward_reader.head

    @property
    def backward_head(self):
        if self.backward_reader is None:
            return None
        return self.backward_reader.head

    