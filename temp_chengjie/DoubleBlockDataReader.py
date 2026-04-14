from BlockDataReader import BlockDataReader

from mossbauer import mossbauer
from sql import sql


class DoubleBlockDataReader:
    def __init__(self,
                 db=None,
                 reader_cls=BlockDataReader,
                 pattern="*.dat.*",
                 table="science_run1",
                 data_tz="America/Los_Angeles"):
        
        self.db = sql() if db is None else db
        self.reader_cls = reader_cls

        self.driver_cls = mossbauer
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

    