import numpy as np
from base_driver import BaseFrameDriver


class L2ParaDriver(BaseFrameDriver):
    """
    Reader for L2Para side-stream files.

    L2Para is the 122keV which defines from 100keV to 140keV;

    40Bytes Head; 
    22*96 uint32 Data;
    Compression Ratio 200, similar to L3 data; 
    """

    BY, BX = 22, 96
    NBLOCK = BY * BX                        # 2112

    DATA_BYTES = NBLOCK * 4                 # uint32
    FRAME_BYTES = BaseFrameDriver.HEAD_BYTES + DATA_BYTES   # 8488
    FRAME_U16 = FRAME_BYTES // 2            # 4244
    DATA_U16 = DATA_BYTES // 2              # 4224

    def __init__(self, fname):
        super().__init__(fname)
        self._data = None

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

    def get_img(self, i):
        return self.load_data()[i]
