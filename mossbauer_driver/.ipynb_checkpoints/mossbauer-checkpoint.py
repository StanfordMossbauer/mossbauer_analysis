import os
import struct
import numpy as np
from datetime import datetime, timedelta


class mossbauer:
    """
    Reader for L3 files.

    Per frame layout:
      8B   SSI
      32B  original header
      8448 x uint16 image (44 x 192)

    Combined head is treated as 40B:
      - word0, word1   : SSI
      - word2 ... word9: original 32B header words
    """

    # -----------------------------
    # fixed L3 geometry
    # -----------------------------
    SSI_BYTES   = 8
    HEAD0_BYTES = 32
    HEAD_BYTES  = SSI_BYTES + HEAD0_BYTES   # 40 bytes

    SSI_U16   = SSI_BYTES // 2              # 4
    HEAD0_U16 = HEAD0_BYTES // 2            # 16
    HEAD_U16  = HEAD_BYTES // 2             # 20

    SSI_U32   = SSI_BYTES // 4              # 2
    HEAD0_U32 = HEAD0_BYTES // 4            # 8
    HEAD_U32  = HEAD_BYTES // 4             # 10

    BY, BX = 44, 192
    NPIX = BY * BX                          # 8448

    DATA_BYTES = NPIX * 2                   # uint16
    FRAME_BYTES = HEAD_BYTES + DATA_BYTES   # 16936
    FRAME_U16 = FRAME_BYTES // 2            # 8468
    DATA_U16 = DATA_BYTES // 2              # 8448

    def __init__(self, fname: str):
        self.fname = fname

        self._buf = None
        self._blk = None
        self._head_u16 = None
        self._head_u32 = None
        self._data = None
        self._nframes = None

    # ------------------------------------------------------------------
    # base load
    # ------------------------------------------------------------------
    def _load_base(self):
        if self._blk is not None:
            return

        byte_size = os.path.getsize(self.fname)
        if byte_size % self.FRAME_BYTES != 0:
            raise RuntimeError(
                f"File length incorrect: {byte_size} is not a multiple of frame_bytes={self.FRAME_BYTES}"
            )

        self._nframes = byte_size // self.FRAME_BYTES
        self._buf = np.memmap(self.fname, mode="r", dtype=np.uint16)
        self._blk = self._buf.reshape(self._nframes, self.FRAME_U16)

    # ------------------------------------------------------------------
    # basic properties
    # ------------------------------------------------------------------
    @property
    def nframes(self):
        self._load_base()
        return self._nframes

    # ------------------------------------------------------------------
    # load head / data
    # ------------------------------------------------------------------
    def load_head(self):
        """
        Return combined 40B head as uint32 view with shape (nframes, 10).

        Combined mapping:
          word0, word1   <- SSI
          word2 ... word9 <- original 32B header words
        """
        if self._head_u32 is not None:
            return self._head_u32

        self._load_base()
        self._head_u16 = self._blk[:, :self.HEAD_U16]
        self._head_u32 = self._head_u16.view(np.uint32)
        return self._head_u32

    def load_data(self):
        """
        Return data with shape (nframes, 44, 192), dtype uint16.
        """
        if self._data is not None:
            return self._data

        self._load_base()
        body = self._blk[:, self.HEAD_U16 : self.HEAD_U16 + self.DATA_U16]
        self._data = body.reshape(self._nframes, self.BY, self.BX)
        return self._data
    
    def load_forward_data(self):
        data= self.load_data() 
        direction = self.load_head()[:,6]
        return data[direction==1]
    
    def load_backward_data(self):
        data= self.load_data() 
        direction = self.load_head()[:,6]
        return data[direction==0]

    @staticmethod
    def convert_image(image: np.ndarray) -> np.ndarray:
        row, col4 = image.shape
        col = col4 // 4
        im1 = image[:, 0*col : 1*col]
        im2 = image[:, 1*col : 2*col]
        im3 = image[::-1, 2*col : 3*col]
        im4 = image[::-1, 3*col : 4*col]
        return np.vstack([np.hstack([im3, im4]),
                          np.hstack([im1, im2])])

    # ------------------------------------------------------------------
    # direct getters
    # ------------------------------------------------------------------
    def get_head(self, i: int):
        return self.load_head()[i]

    def get_img(self, i: int):
        return self.load_data()[i]
    
    def get_word(self, j: int):
        """
        Return one column of the combined 40B head.
        """
        return self.load_head()[:, j]

    def get_direction(self):
        return self.load_head()[:,6]

    def summary_head(self, i: int = 0):
        h = self.get_head(i)
        for k, v in enumerate(h):
            print(f"word{k} = {v}")

    # ------------------------------------------------------------------
    # timestamp helpers
    # ------------------------------------------------------------------
    def get_datetime(self, i: int):
        """
        Read timestamp from frame i using the loaded head array.

        In combined head_u32:
          word7 = original header word5 = microseconds
          word9 = original header word7 = seconds
        """
        h = self.get_head(i)
        usec = int(h[7])
        sec = int(h[9])
        return datetime.fromtimestamp(sec) + timedelta(microseconds=usec)





    # Two special functions that accerlate the readout ; 
    # Without load the whole data, just get the result 
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
                raise RuntimeError("file too short to contain first L3 frame header")

        return self._decode_time_from_head40(head40)[0]

    def read_last_time(self):
        """
        Fast path: read only the last complete frame header from disk.
        """
        byte_size = os.path.getsize(self.fname)
        if byte_size < self.FRAME_BYTES:
            raise RuntimeError("file too short to contain one complete L3 frame")
        nframes = byte_size // self.FRAME_BYTES
        if nframes == 0:
            raise RuntimeError("no complete L3 frame found")

        last_offset = (nframes - 1) * self.FRAME_BYTES

        with open(self.fname, "rb") as f:
            f.seek(last_offset)
            head40 = f.read(self.HEAD_BYTES)
            if len(head40) < self.HEAD_BYTES:
                raise RuntimeError("failed to read last L3 frame header")

        return self._decode_time_from_head40(head40)[0]

    def read_time_range(self):
        return (self.read_first_time(), self.read_last_time())