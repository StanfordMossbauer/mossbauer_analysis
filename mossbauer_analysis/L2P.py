import os
import struct
import numpy as np
from datetime import datetime, timedelta


class L2ParaDriver:
    """
    Reader for L2Para side-stream files.

    Per frame layout:
      8B   SSI
      32B  original header
      2112 x uint32 image (22 x 96)

    Combined head is treated as 40B:
      - word0, word1   : SSI
      - word2 ... word9: original 32B header words
    """

    SSI_BYTES   = 8
    HEAD0_BYTES = 32
    HEAD_BYTES  = SSI_BYTES + HEAD0_BYTES   # 40 bytes

    SSI_U16   = SSI_BYTES // 2              # 4
    HEAD0_U16 = HEAD0_BYTES // 2            # 16
    HEAD_U16  = HEAD_BYTES // 2             # 20

    SSI_U32   = SSI_BYTES // 4              # 2
    HEAD0_U32 = HEAD0_BYTES // 4            # 8
    HEAD_U32  = HEAD_BYTES // 4             # 10

    BY, BX = 22, 96
    NBLOCK = BY * BX                        # 2112

    DATA_BYTES = NBLOCK * 4                 # uint32
    FRAME_BYTES = HEAD_BYTES + DATA_BYTES   # 8488
    FRAME_U16 = FRAME_BYTES // 2            # 4244
    DATA_U16 = DATA_BYTES // 2              # 4224

    def __init__(self, fname):
        self.fname = fname
        self._buf = None
        self._blk = None
        self._head_u16 = None
        self._head_u32 = None
        self._data = None
        self._nframes = None

    def _load_base(self):
        if self._blk is not None:
            return

        u16_size = os.path.getsize(self.fname) // 2
        if u16_size % self.FRAME_U16 != 0:
            raise RuntimeError("File length is not an integer number of L2Para frames")

        self._nframes = u16_size // self.FRAME_U16
        self._buf = np.memmap(self.fname, mode="r", dtype=np.uint16)
        self._blk = self._buf.reshape(self._nframes, self.FRAME_U16)

    @property
    def nframes(self):
        self._load_base()
        return self._nframes

    def load_head(self):
        """
        Return combined 40B head as uint32 view with shape (nframes, 10).
        """
        if self._head_u32 is not None:
            return self._head_u32

        self._load_base()
        self._head_u16 = self._blk[:, :self.HEAD_U16]
        self._head_u32 = self._head_u16.view(np.uint32)
        return self._head_u32

    def load_data(self):
        """
        Return data with shape (nframes, 22, 96), dtype uint32.
        """
        if self._data is not None:
            return self._data

        self._load_base()
        body_u16 = self._blk[:, self.HEAD_U16 : self.HEAD_U16 + self.DATA_U16]
        body_u32 = body_u16.view(np.uint32)
        self._data = body_u32.reshape(self._nframes, self.BY, self.BX)
        return self._data

    def get_head(self, i):
        return self.load_head()[i]

    def get_img(self, i):
        return self.load_data()[i]

    def get_word(self, j):
        return self.load_head()[:, j]

    def summary_head(self, i=0):
        h = self.get_head(i)
        for k, v in enumerate(h):
            print(f"word{k} = {v}")

    def get_datetime(self, i):
        """
        In the combined 40B head:
          word7 = original header word5 = microseconds
          word9 = original header word7 = seconds
        """
        h = self.get_head(i)
        usec = int(h[7])
        sec = int(h[9])
        return datetime.fromtimestamp(sec) + timedelta(microseconds=usec)

    # ------------------------------------------------------------------
    # fast timestamp helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _decode_time_from_head40(head40: bytes):
        """
        In the combined 40B head:
          byte 28~31 : original header word5 = microseconds
          byte 36~39 : original header word7 = seconds
        """
        if len(head40) < 40:
            raise RuntimeError("head40 too short")

        usec = struct.unpack_from("<I", head40, 28)[0]
        sec  = struct.unpack_from("<I", head40, 36)[0]
        dt = datetime.fromtimestamp(sec) + timedelta(microseconds=usec)
        return dt, sec, usec

    def read_first_time(self):
        """
        Fast path: read only the first frame header from disk.
        """
        with open(self.fname, "rb") as f:
            head40 = f.read(self.HEAD_BYTES)
            if len(head40) < self.HEAD_BYTES:
                raise RuntimeError("file too short to contain first L2Para frame header")

        return self._decode_time_from_head40(head40)[0]

    def read_last_time(self):
        """
        Fast path: read only the last complete frame header from disk.
        """
        byte_size = os.path.getsize(self.fname)
        if byte_size < self.FRAME_BYTES:
            raise RuntimeError("file too short to contain one complete L2Para frame")

        nframes = byte_size // self.FRAME_BYTES
        if nframes == 0:
            raise RuntimeError("no complete L2Para frame found")

        last_offset = (nframes - 1) * self.FRAME_BYTES

        with open(self.fname, "rb") as f:
            f.seek(last_offset)
            head40 = f.read(self.HEAD_BYTES)
            if len(head40) < self.HEAD_BYTES:
                raise RuntimeError("failed to read last L2Para frame header")

        return self._decode_time_from_head40(head40)[0]

    def read_time_range(self):
        return (self.read_first_time(), self.read_last_time())