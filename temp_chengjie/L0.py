import numpy as np
from base_driver import BaseFrameDriver


class L0Driver(BaseFrameDriver):
    """
    Reader for L0 files.

    Per frame layout:
      8B   SSI
      32B  original header
      176 x 768 uint16 image

    Combined head is treated as 40B:
      - word0, word1   : SSI
      - word2 ... word9: original 32B header words
    """

    NY, NX = 176, 768
    NPIX = NY * NX

    DATA_BYTES = NPIX * 2                   # uint16
    TAIL_BYTES = 4620
    FRAME_BYTES = BaseFrameDriver.HEAD_BYTES + DATA_BYTES + TAIL_BYTES   # 274996
    FRAME_U16 = FRAME_BYTES // 2            # 137498
    DATA_U16 = DATA_BYTES // 2              # 135168

    def __init__(self, fname):
        super().__init__(fname)
        self._data = None

    def load_data(self):
        """
        Return image data with shape (nframes, 176, 768), dtype uint16.
        """
        if self._data is not None:
            return self._data

        self._load_base()
        body = self._blk[:, self.HEAD_U16 : self.HEAD_U16 + self.DATA_U16]
        self._data = body.reshape(self._nframes, self.NY, self.NX)
        return self._data

    def get_img(self, i):
        return self.load_data()[i]
