# This is the main class for the data; 

# This also includes the basic data analysis functions including the last side and right side; 


# An important ; 


# importantly is to have a manual ; 

import os
import numpy as np


class L3Driver:
    """
    L3 fixed-frame reader using memmap.

    Per frame layout:
      8B   SSI
      32B  original header
      8448 x uint16 image

    We expose a combined 40B head:
      - head_bytes: raw 40-byte head
      - head_u32  : 10 x uint32 view of that 40-byte head

    In this combined head_u32:
      word0, word1   <- SSI (8 bytes)
      word2 ... word9 <- original 32B header words
    """

    SSI_U16   = 4          # 8 bytes
    HEAD0_U16 = 16         # original 32 bytes
    HEAD_U16  = SSI_U16 + HEAD0_U16   # 20 uint16 = 40 bytes

    BY, BX = 44, 192
    NPIX = BY * BX

    FRAME_U16 = HEAD_U16 + NPIX   # 20 + 8448

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
            raise RuntimeError("File length is not an integer number of L3 frames")

        self._nframes = u16_size // self.FRAME_U16
        self._buf = np.memmap(self.fname, mode="r", dtype=np.uint16)
        self._blk = self._buf.reshape(self._nframes, self.FRAME_U16)

    @property
    def nframes(self):
        self._load_base()
        return self._nframes

    def load_head(self):
        """
        Load combined 40B head as:
          - _head_u16 : shape (nframes, 20)
          - _head_u32 : shape (nframes, 10)
        """
        if self._head_u32 is not None:
            return self._head_u32

        self._load_base()
        self._head_u16 = self._blk[:, :self.HEAD_U16]
        self._head_u32 = self._head_u16.view(np.uint32)
        return self._head_u32

    def load_data(self):
        if self._data is not None:
            return self._data

        self._load_base()
        body = self._blk[:, self.HEAD_U16 : self.HEAD_U16 + self.NPIX]
        self._data = body.reshape(self._nframes, self.BY, self.BX)
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