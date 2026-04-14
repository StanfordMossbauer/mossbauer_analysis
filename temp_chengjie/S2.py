import numpy as np
from base_driver import BaseFrameDriver


class L2SpectrumDriver(BaseFrameDriver):
    """
    Reader for L2Spectrum side-stream files.

    Per frame layout:
      8B   SSI
      32B  original header
      1460 x uint32 spectrum bins

    Combined head is treated as 40B:
      - word0, word1   : SSI
      - word2 ... word9: original 32B header words
    """

    NBINS = 1460

    DATA_BYTES = NBINS * 4                  # uint32
    FRAME_BYTES = BaseFrameDriver.HEAD_BYTES + DATA_BYTES   # 5880
    FRAME_U16 = FRAME_BYTES // 2            # 2940
    DATA_U16 = DATA_BYTES // 2              # 2920

    def __init__(self, fname):
        super().__init__(fname)
        self._data = None
        self.energy = np.append(np.arange(1280) / 64, np.arange(20, 200))

    def load_data(self):
        """
        Return spectrum data with shape (nframes, 1460), dtype uint32.
        """
        if self._data is not None:
            return self._data

        self._load_base()
        body_u16 = self._blk[:, self.HEAD_U16 : self.HEAD_U16 + self.DATA_U16]
        self._data = body_u16.view(np.uint32)
        return self._data

    def get_energy(self):
        self.energy = np.append(np.arange(1280) / 64, np.arange(20, 200))
        return self.energy

    def get_small_spec(self, i):
        return self.get_energy()[:1280], self.get_spec(i)[:1280]

    def get_long_spec(self, i):
        return self.get_energy()[1280:], self.get_spec(i)[1280:]

    def get_spec(self, i):
        return self.load_data()[i]
