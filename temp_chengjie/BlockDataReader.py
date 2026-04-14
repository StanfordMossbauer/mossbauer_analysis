import numpy as np
from datetime import timezone

from block import get_true_time_range
from file_finder import find_files_in_range

from sql import sql

from mossbauer import mossbauer
from L0 import L0Driver
from L2P import L2ParaDriver
from S2 import L2SpectrumDriver
from raw import epix


class BlockDataReader:
    def __init__(self,
                 db=None,
                 driver_cls=mossbauer,
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