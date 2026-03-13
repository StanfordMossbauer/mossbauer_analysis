# This is the driver to read out the raw data.
# It is close to the direct camera output, with added timestamps in the header.

import os
import re
import struct
import numpy as np
import bottleneck as bn
from matplotlib import pyplot as plt
from datetime import datetime, timedelta


class epix:
    """
    Raw data driver + processing helper.

    Per frame layout:
      8B   SSI
      32B  original header
      main image body
      tail area

    Combined head is treated as 40B:
      - word0, word1   : SSI
      - word2 ... word9: original 32B header words

    Direction:
      - original header word4
      - combined head word6
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

    def __init__(self,
                 fname: str,
                 frame_bytes: int = 274996,
                 nrow: int = 176,
                 ncolumn: int = 192,
                 frequency: float = 1.0,
                 mean_gain: float = 17.0
                 ) -> None:

        self.fname = fname
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
            raise ValueError("frame_bytes too small for declared geometry")

        self._buf = None
        self._blk = None
        self._head_u16 = None
        self._head_u32 = None
        self._data = None
        self._tail = None
        self._direction = None
        self._forward_mask = None
        self._backward_mask = None
        self._nframes = None

        self._t0 = self._parse_timestamp()

    # ------------------------------------------------------------------
    # filename timestamp
    # ------------------------------------------------------------------
    def _parse_timestamp(self):
        m = re.search(r"(\d{8})_(\d{6})", os.path.basename(self.fname))
        if not m:
            return None
        date_part, time_part = m.groups()
        iso = (f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}T"
               f"{time_part[:2]}:{time_part[2:4]}:{time_part[4:]}")
        return np.datetime64(iso, "s")

    # ------------------------------------------------------------------
    # base load
    # ------------------------------------------------------------------
    def _load_base(self):
        if self._blk is not None:
            return

        byte_size = os.path.getsize(self.fname)
        if byte_size % self.frame_bytes != 0:
            raise RuntimeError(
                f"File length incorrect: {byte_size} is not a multiple of frame_bytes={self.frame_bytes}"
            )

        self._nframes = byte_size // self.frame_bytes
        self._buf = np.memmap(self.fname, mode="r", dtype=np.uint16)
        self._blk = self._buf.reshape(self._nframes, self.frame_u16)

    def _load_direction_mask(self):
        if self._direction is not None:
            return
        self._direction = self.load_head()[:, 6]   # combined word6 = original word4
        self._forward_mask = (self._direction == 1)
        self._backward_mask = (self._direction == 0)

    # ------------------------------------------------------------------
    # load head / data / tail
    # ------------------------------------------------------------------
    def load_head(self):
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

    @property
    def nframes(self):
        self._load_base()
        return self._nframes

    # ------------------------------------------------------------------
    # direct getters
    # ------------------------------------------------------------------
    def get_head(self, i: int):
        return self.load_head()[i]

    def get_img(self, i: int):
        return self.load_data()[i]

    def get_tail(self, i: int):
        return self.load_tail()[i]

    def get_word(self, j: int):
        return self.load_head()[:, j]

    def get_direction(self):
        # combined word6 = original header word4
        return self.load_head()[:, 6]

    def summary_head(self, i: int = 0):
        h = self.get_head(i)
        for k, v in enumerate(h):
            print(f"word{k} = {v}")

    def frame(self, i: int, n: int = 1):
        if n == 1:
            return self.data[i]
        return self.data[i:i+n]

    # ------------------------------------------------------------------
    # time
    # ------------------------------------------------------------------
    def get_datetime(self, i: int):
        h = self.get_head(i)
        usec = int(h[7])   # combined word7 = original word5
        sec = int(h[9])    # combined word9 = original word7
        return datetime.fromtimestamp(sec) + timedelta(microseconds=usec)

    @staticmethod
    def _decode_time_from_head40(head40: bytes):
        if len(head40) < 40:
            raise RuntimeError("head40 too short")

        usec = struct.unpack_from("<I", head40, 28)[0]
        sec  = struct.unpack_from("<I", head40, 36)[0]
        dt = datetime.fromtimestamp(sec) + timedelta(microseconds=usec)
        return dt, sec, usec

    def read_first_time(self):
        with open(self.fname, "rb") as f:
            head40 = f.read(self.HEAD_BYTES)
            if len(head40) < self.HEAD_BYTES:
                raise RuntimeError("file too short to contain first raw frame header")
        return self._decode_time_from_head40(head40)[0]

    def read_last_time(self):
        byte_size = os.path.getsize(self.fname)
        if byte_size < self.frame_bytes:
            raise RuntimeError("file too short to contain one complete raw frame")

        nframes = byte_size // self.frame_bytes
        if nframes == 0:
            raise RuntimeError("no complete raw frame found")

        last_offset = (nframes - 1) * self.frame_bytes

        with open(self.fname, "rb") as f:
            f.seek(last_offset)
            head40 = f.read(self.HEAD_BYTES)
            if len(head40) < self.HEAD_BYTES:
                raise RuntimeError("failed to read last raw frame header")

        return self._decode_time_from_head40(head40)[0]

    def read_time_range(self):
        return (self.read_first_time(), self.read_last_time())

    def _time_axis(self):
        N = self.nframes
        if self._t0 is None:
            return np.arange(N, dtype=float) * self.period_s

        if self.period_s.is_integer():
            step = np.timedelta64(int(self.period_s), "s")
            return np.arange(self._t0,
                             self._t0 + step * N,
                             step,
                             dtype="datetime64[s]")

        step_ns = int(round(self.period_s * 1e9))
        base = self._t0.astype("datetime64[ns]")
        return base + np.arange(N, dtype="timedelta64[ns]") * step_ns

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
                    f"gain shape must be ({self.nrow}, {4*self.ncolumn}), got {gain.shape}"
                )
            frame /= gain

        if mask is None:
            return frame

        if mask.shape != (self.nrow, 4 * self.ncolumn):
            raise ValueError(
                f"mask shape must be ({self.nrow}, {4*self.ncolumn}), got {mask.shape}"
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
                       title: str = "Bad Pixels",
                       point_size: int = 3,
                       color: str = "red",
                       xlabel: str = "Column",
                       ylabel: str = "Row",
                       figsize=(7, 5),
                       invert_y: bool = True) -> None:

        mask = epix.convert_image(raw_mask)
        ny, nx = mask.shape
        Y, X = np.mgrid[0:ny, 0:nx]

        x_bad = X[mask].ravel()
        y_bad = Y[mask].ravel()
        print(f"bad pixels: {len(x_bad)}")

        plt.figure(figsize=figsize)
        plt.title(title)
        plt.scatter(x_bad, y_bad, s=point_size, color=color)

        if invert_y:
            plt.gca().invert_yaxis()

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.show()