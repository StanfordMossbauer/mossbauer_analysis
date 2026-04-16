import numpy as np
from .base_driver import BaseFrameDriver


# This is the driver for a single mossbauer file; 
class L3Driver(BaseFrameDriver):
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

    BY, BX = 44, 192
    NPIX = BY * BX                          # 8448

    DATA_BYTES = NPIX * 2                   # uint16
    FRAME_BYTES = BaseFrameDriver.HEAD_BYTES + DATA_BYTES   # 16936
    FRAME_U16 = FRAME_BYTES // 2            # 8468
    DATA_U16 = DATA_BYTES // 2              # 8448

    def __init__(self, fname: str):
        super().__init__(fname)
        self._data = None

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
        data = self.load_data()
        direction = self.load_head()[:, 6]
        return data[direction == 1]
    
    def load_backward_data(self):
        data = self.load_data()
        direction = self.load_head()[:, 6]
        return data[direction == 0]

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

    def get_img(self, i: int):
        return self.load_data()[i]

    def get_direction(self):
        return self.load_head()[:, 6]

    def get_daq_count(self):
        return self.load_head()[:, 3]

    # Here we record the run ID of the first frame that is being compressed and the last frame that is being compressed; 
    # We shall see that the first frame ID has a difference of 1 between moving forward and backward, it could be 1 or -1 
    # depends on the initial direction of the acquision  ; 
    # For a compression ratio of 200, we shall see a difference of (200-1)*2 between the first frame and the last frame , and a difference of 400 between two groups of compressed frames; 
    def get_initial_run_count(self):
        return self.load_head()[:, 4]

    def get_last_run_count(self):
        return self.load_head()[:, 8]

    def get_compressed_counts(self):
        return (self.load_head()[:, 8] - self.load_head()[:, 4]) // 2 + 1

    def get_single_datetime(self, i: int):
        return self.get_datetime(i)
