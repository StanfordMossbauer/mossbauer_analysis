# This is the driver to read out the raw data.
# It is close to the direct camera output, with added timestamps in the header.

import os
import re
import numpy as np
import bottleneck as bn
from matplotlib import pyplot as plt
from base_driver import BaseFrameDriver


class epix(BaseFrameDriver):
    """
    Raw data driver + processing helper.


    1% 4Hz, without any signal processing;
    40 Bytes Head
    274956 Bytes of Data; 
    
    """

    def __init__(self,
                 fname: str,
                 frame_bytes: int = 274996,
                 nrow: int = 176,
                 ncolumn: int = 192,
                 frequency: float = 1.0,
                 mean_gain: float = 17.0
                 ) -> None:

        super().__init__(fname)
        self.nrow = int(nrow)
        self.ncolumn = int(ncolumn)
        self.npix = 4 * self.nrow * self.ncolumn   # 176 x 768
        self.frame_bytes = int(frame_bytes)
        self.frame_u16 = self.frame_bytes // 2

        self.frequency = float(frequency)
        self.period_s = 1.0 / self.frequency
        self.mean_gain = float(mean_gain)

        self.tail_u16 = self.frame_u16 - self.HEAD_U16 - self.npix
        if self.tail_u16 < 0:
            raise ValueError('frame_bytes too small for declared geometry')

        self._data = None
        self._tail = None
        self._direction = None
        self._forward_mask = None
        self._backward_mask = None

        self._t0 = self._parse_timestamp()

    # ------------------------------------------------------------------
    # filename timestamp
    # ------------------------------------------------------------------
    def _parse_timestamp(self):
        m = re.search(r'(\d{8})_(\d{6})', os.path.basename(self.fname))
        if not m:
            return None
        date_part, time_part = m.groups()
        iso = (f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}T"
               f"{time_part[:2]}:{time_part[2:4]}:{time_part[4:]}")
        return np.datetime64(iso, 's')

    def _load_direction_mask(self):
        if self._direction is not None:
            return
        self._direction = self.load_head()[:, 6]   # combined word6 = original word4
        self._forward_mask = (self._direction == 1)
        self._backward_mask = (self._direction == 0)

    # ------------------------------------------------------------------
    # load head / data / tail
    # ------------------------------------------------------------------
    def load_data(self):
        if self._data is not None:
            return self._data

        self._load_base()
        body = self._blk[:, self.HEAD_U16 : self.HEAD_U16 + self.npix]
        self._data = body.reshape(self._nframes, self.nrow, 4 * self.ncolumn)
        return self._data

    def load_tail(self):
        if self._tail is not None:
            return self._tail

        self._load_base()
        start = self.HEAD_U16 + self.npix
        stop = start + self.tail_u16
        self._tail = self._blk[:, start:stop]
        return self._tail

    def load_forward_data(self):
        self._load_direction_mask()
        return self.load_data()[self._forward_mask]

    def load_backward_data(self):
        self._load_direction_mask()
        return self.load_data()[self._backward_mask]

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------
    @property
    def data(self):
        return self.load_data()

    @property
    def head(self):
        return self.load_head()

    # ------------------------------------------------------------------
    # direct getters
    # ------------------------------------------------------------------
    def get_img(self, i: int):
        return self.load_data()[i]

    def get_tail(self, i: int):
        return self.load_tail()[i]

    def get_direction(self):
        # combined word6 = original header word4
        return self.load_head()[:, 6]

    def frame(self, i: int, n: int = 1):
        if n == 1:
            return self.data[i]
        return self.data[i:i+n]

    # ------------------------------------------------------------------
    # time
    # ------------------------------------------------------------------
    def _time_axis(self):
        N = self.nframes
        if self._t0 is None:
            return np.arange(N, dtype=float) * self.period_s

        if self.period_s.is_integer():
            step = np.timedelta64(int(self.period_s), 's')
            return np.arange(self._t0,
                             self._t0 + step * N,
                             step,
                             dtype='datetime64[s]')

        step_ns = int(round(self.period_s * 1e9))
        base = self._t0.astype('datetime64[ns]')
        return base + np.arange(N, dtype='timedelta64[ns]') * step_ns

    def time_series(self):
        return self._time_axis()

    # ------------------------------------------------------------------
    # quick summaries
    # ------------------------------------------------------------------
    def frame_mean(self):
        return self.data.mean(axis=(1, 2))

    def mean_series(self):
        return self.frame_mean()

    def pixel_series(self, x: int, y: int):
        return self.data[:, x, y]

    def pixel_series_p(self, x: int, y: int, p: int):
        return self.data[:, x, y + p * self.ncolumn]

    # ------------------------------------------------------------------
    # dark/std maps
    # ------------------------------------------------------------------
    def low_occupy_dark(self):
        return bn.nanmedian(self.data[0:2000], axis=0)

    def low_occupy_std(self):
        return bn.nanstd(self.data[0:2000], axis=0)

    def get_dark(self):
        return bn.nanmedian(self.data, axis=0)

    def get_std(self):
        return bn.nanstd(self.data, axis=0)

    # ------------------------------------------------------------------
    # image conversion
    # ------------------------------------------------------------------
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
    # common-mode helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _frame_common_mode(frame: np.ndarray, n0: int) -> None:
        idx = frame.ravel() < 4 * n0
        if np.any(idx):
            frame -= np.median(frame.ravel()[idx])

    @staticmethod
    def _col_common_mode(frame: np.ndarray, n1: int) -> None:
        ny, nx = frame.shape
        for icol in range(nx):
            col = frame[:, icol]
            idx = col < 4 * n1
            if np.any(idx):
                col -= np.median(col[idx])

    # ------------------------------------------------------------------
    # single-frame preprocessing
    # ------------------------------------------------------------------
    def _prep_frame(self,
                    iframe: int,
                    dark: np.ndarray,
                    n0: int,
                    n1: int,
                    col_cm: bool = True,
                    mask: np.ndarray | None = None,
                    gain: float | np.ndarray | None = None
                   ) -> np.ndarray:

        frame = self.frame(iframe).astype(float, copy=True)
        frame -= dark
        self._frame_common_mode(frame, n0)
        if col_cm:
            self._col_common_mode(frame, n1)

        if gain is None:
            frame /= self.mean_gain
        elif np.isscalar(gain):
            frame /= float(gain)
        else:
            if gain.shape != (self.nrow, 4 * self.ncolumn):
                raise ValueError(
                    f'gain shape must be ({self.nrow}, {4*self.ncolumn}), got {gain.shape}'
                )
            frame /= gain

        if mask is None:
            return frame

        if mask.shape != (self.nrow, 4 * self.ncolumn):
            raise ValueError(
                f'mask shape must be ({self.nrow}, {4*self.ncolumn}), got {mask.shape}'
            )
        return frame * mask

    def background(self,
                   iframe: int,
                   dark: np.ndarray,
                   n0: int,
                   n1: int,
                   mask: np.ndarray | None = None,
                   gain: float | np.ndarray | None = None):
        return self._prep_frame(iframe, dark, n0, n1, col_cm=True, mask=mask, gain=gain)

    def histogram(self,
                  iframe: int,
                  dark: np.ndarray,
                  n0: int,
                  n1: int,
                  bins,
                  mask: np.ndarray | None = None,
                  gain: float | np.ndarray | None = None):
        frame = self._prep_frame(iframe, dark, n0, n1, col_cm=True, mask=mask, gain=gain)
        return np.histogram(frame.ravel(), bins=bins)[0]

    def distribution(self,
                     iframe: int,
                     dark: np.ndarray,
                     n0: int,
                     n1: int,
                     bins,
                     mask: np.ndarray | None = None,
                     gain: float | np.ndarray | None = None):
        frame = self._prep_frame(iframe, dark, n0, n1, col_cm=True, mask=mask, gain=gain)
        return (frame > 10).astype(int)

    def dis_histogram(self,
                      iframe: int,
                      dark: np.ndarray,
                      n0: int,
                      n1: int,
                      bins,
                      mask: np.ndarray | None = None,
                      gain: float | np.ndarray | None = None):
        frame = self._prep_frame(iframe, dark, n0, n1, col_cm=True, mask=mask, gain=gain)
        return np.histogram(frame.ravel(), bins=bins)[0], (frame > 10).astype(int)

    # ------------------------------------------------------------------
    # charge sharing
    # ------------------------------------------------------------------
    @staticmethod
    def _centroid_filter(frame: np.ndarray, n2: int):
        ny, nx = frame.shape
        as1 = frame * (frame > 4 * n2)

        up = np.zeros_like(as1)
        down = np.zeros_like(as1)
        left = np.zeros_like(as1)
        right = np.zeros_like(as1)

        up[0:ny - 1, :] = as1[1:, :]
        down[1:, :] = as1[0:ny - 1, :]
        left[:, 0:nx - 1] = as1[:, 1:nx]
        right[:, 1:nx] = as1[:, 0:nx - 1]

        cen_v = as1 + up + down + left + right
        cen_f = (as1 > up) & (as1 > down) & (as1 > left) & (as1 > right)
        return cen_v * cen_f

    def cs_noise(self,
                 iframe: int,
                 dark: np.ndarray,
                 n0: int,
                 n1: int,
                 n2: int,
                 mask: np.ndarray | None = None,
                 gain: float | np.ndarray | None = None):
        frame = self._prep_frame(iframe, dark, n0, n1, col_cm=True, mask=mask, gain=gain)
        return self._centroid_filter(frame, n2)

    def cs_histogram(self,
                     iframe: int,
                     dark: np.ndarray,
                     n0: int,
                     n1: int,
                     n2: int,
                     bins,
                     mask: np.ndarray | None = None,
                     gain: float | np.ndarray | None = None):
        frame = self.cs_noise(iframe, dark, n0, n1, n2, mask=mask, gain=gain)
        return np.histogram(frame.ravel(), bins=bins)[0]

    # ------------------------------------------------------------------
    # bad-pixel scatter
    # ------------------------------------------------------------------
    @staticmethod
    def binary_scatter(raw_mask: np.ndarray,
                       *,
                       title: str = 'Bad Pixels',
                       point_size: int = 3,
                       color: str = 'red',
                       xlabel: str = 'Column',
                       ylabel: str = 'Row',
                       figsize=(7, 5),
                       invert_y: bool = True) -> None:

        mask = epix.convert_image(raw_mask)
        ny, nx = mask.shape
        Y, X = np.mgrid[0:ny, 0:nx]

        x_bad = X[mask].ravel()
        y_bad = Y[mask].ravel()
        print(f'bad pixels: {len(x_bad)}')

        plt.figure(figsize=figsize)
        plt.title(title)
        plt.scatter(x_bad, y_bad, s=point_size, color=color)

        if invert_y:
            plt.gca().invert_yaxis()

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.show()
