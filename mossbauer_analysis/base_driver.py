from datetime import datetime, timedelta, timezone
import os
import struct
import numpy as np

UTC = timezone.utc

class BaseFrameDriver:
    """
    Common base class for frame-based binary stream files with a 40-byte
    combined head = 8-byte SSI + 32-byte original header.

    Time convention:
      - header seconds are interpreted as UTC epoch seconds
      - all datetime helpers return UTC-aware datetime objects
      - datetime64 helpers are interpreted as UTC
    """

    SSI_BYTES = 8
    HEAD0_BYTES = 32
    HEAD_BYTES = SSI_BYTES + HEAD0_BYTES

    SSI_U16 = SSI_BYTES // 2
    HEAD0_U16 = HEAD0_BYTES // 2
    HEAD_U16 = HEAD_BYTES // 2

    SSI_U32 = SSI_BYTES // 4
    HEAD0_U32 = HEAD0_BYTES // 4
    HEAD_U32 = HEAD_BYTES // 4

    def __init__(self, fname: str):
        self.fname = fname
        self._buf = None
        self._blk = None
        self._head_u16 = None
        self._head_u32 = None
        self._nframes = None

    def _get_frame_bytes(self):
        if hasattr(self, 'frame_bytes'):
            return int(self.frame_bytes)
        return int(self.FRAME_BYTES)

    def _get_frame_u16(self):
        if hasattr(self, 'frame_u16'):
            return int(self.frame_u16)
        return int(self.FRAME_U16)

    def _load_base(self):
        if self._blk is not None:
            return

        frame_bytes = self._get_frame_bytes()
        frame_u16 = self._get_frame_u16()
        byte_size = os.path.getsize(self.fname)
        if byte_size % frame_bytes != 0:
            raise RuntimeError(
                f"File length incorrect: {byte_size} is not a multiple of frame_bytes={frame_bytes}"
            )

        self._nframes = byte_size // frame_bytes
        self._buf = np.memmap(self.fname, mode='r', dtype='<u2')
        self._blk = self._buf.reshape(self._nframes, frame_u16)

    @property
    def nframes(self):
        self._load_base()
        return self._nframes

    def load_head(self):
        if self._head_u32 is not None:
            return self._head_u32

        self._load_base()
        self._head_u16 = self._blk[:, :self.HEAD_U16]
        self._head_u32 = self._head_u16.view(np.uint32)
        return self._head_u32

    def get_head(self, i: int):
        return self.load_head()[i]

    def get_word(self, j: int):
        return self.load_head()[:, j]

    def summary_head(self, i: int = 0):
        h = self.get_head(i)
        for k, v in enumerate(h):
            print(f"word{k} = {v}")

    def get_single_datetime(self, i: int):
        h = self.get_head(i)
        usec = int(h[7])
        sec = int(h[9])
        return datetime.fromtimestamp(sec, tz=UTC) + timedelta(microseconds=usec)

    def get_all_datetime64(self):
        h = self.load_head()
        usec = h[:, 7].astype(np.int64)
        sec = h[:, 9].astype(np.int64)
        return sec.astype('datetime64[s]') + usec.astype('timedelta64[us]')

    def get_all_datetime(self):
        return [t.astype(object).replace(tzinfo=UTC) for t in self.get_all_datetime64()]

    @staticmethod
    def _decode_time_from_head40(head40: bytes):
        if len(head40) < 40:
            raise RuntimeError('head40 too short')

        usec = struct.unpack_from('<I', head40, 28)[0]
        sec = struct.unpack_from('<I', head40, 36)[0]
        dt = datetime.fromtimestamp(sec, tz=UTC) + timedelta(microseconds=usec)
        return dt, sec, usec

    def read_first_time(self):
        with open(self.fname, 'rb') as f:
            head40 = f.read(self.HEAD_BYTES)
            if len(head40) < self.HEAD_BYTES:
                raise RuntimeError('file too short to contain first frame header')
        return self._decode_time_from_head40(head40)[0]
    get_first_time=read_first_time
    
    def read_last_time(self):
        frame_bytes = self._get_frame_bytes()
        byte_size = os.path.getsize(self.fname)
        if byte_size < frame_bytes:
            raise RuntimeError('file too short to contain one complete frame')

        nframes = byte_size // frame_bytes
        if nframes == 0:
            raise RuntimeError('no complete frame found')

        last_offset = (nframes - 1) * frame_bytes

        with open(self.fname, 'rb') as f:
            f.seek(last_offset)
            head40 = f.read(self.HEAD_BYTES)
            if len(head40) < self.HEAD_BYTES:
                raise RuntimeError('failed to read last frame header')

        return self._decode_time_from_head40(head40)[0]
    get_last_time=read_last_time
    
    def read_time_range(self):
        return self.read_first_time(), self.read_last_time()
    get_time_range=read_time_range
    
    
    def convert_image(image: np.ndarray) -> np.ndarray:
        row, col4 = image.shape
        col = col4 // 4
        im0 = image[:, 0*col : 1*col]
        im1 = image[:, 1*col : 2*col]
        im2 = image[::-1, 2*col : 3*col]
        im3 = image[::-1, 3*col : 4*col]
        return np.vstack([np.hstack([im2, im3])
                          ,np.hstack([im0, im1])])
                        ##########################
                        ##### Camera Body ########
                        ##########################
                        ##########################
                        ##########################

    