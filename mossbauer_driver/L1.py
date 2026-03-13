
# The l1 still contains all the information and does not include the charge sharing; 
# This L1 is very big, so ideally we do not want to read this file; 
# But if necessary, this file could be very useful; 
# But to read it out is a painful process; 


import os
import struct
import numpy as np
from datetime import datetime, timedelta


class L1BMDriver:
    """
    Reader for variable-length L1 bitmask-compressed files.

    Per frame layout:
      8B   SSI
      32B  original header
      E1 header:
           magic(4s), ny(u16), nx(u16), thr(u16), rsv(u16), count(u32), mbytes(u32)
      4B   gap/padding
      mbytes   packed bitmask
      count*2  uint16 values

    Public style is kept similar to fixed-frame drivers:
      - nframes
      - get_head(i)
      - get_datetime(i)
      - read_first_time()
      - read_last_time()
      - get_mask(i)
      - get_vals(i)
    """

    E1_FMT = "<4sHHHHII"
    E1_SZ = struct.calcsize(E1_FMT)

    HEAD40_BYTES = 40   # 8B SSI + 32B orig32

    def __init__(self, fname: str):
        self.fname = fname

        self._index = None
        self._nframes = None

    # ------------------------------------------------------------------
    # build frame index
    # ------------------------------------------------------------------
    def _build_index(self):
        if self._index is not None:
            return

        idx = []
        offset = 0
        fsize = os.path.getsize(self.fname)

        with open(self.fname, "rb") as f:
            while True:
                frame_start = offset

                ssi = f.read(8)
                if not ssi:
                    break
                if len(ssi) < 8:
                    break
                offset += 8

                orig32 = f.read(32)
                if len(orig32) < 32:
                    break
                offset += 32

                e1hdr = f.read(self.E1_SZ)
                if len(e1hdr) < self.E1_SZ:
                    break
                offset += self.E1_SZ

                magic, ny, nx, thr, _rsv, count, mbytes = struct.unpack(self.E1_FMT, e1hdr)
                if magic != b"E1BM":
                    break

                gap4 = f.read(4)
                if len(gap4) < 4:
                    break
                offset += 4

                mask_offset = offset
                mask_bytes = f.read(mbytes)
                if len(mask_bytes) < mbytes:
                    break
                offset += mbytes

                vals_offset = offset
                vals_bytes = f.read(count * 2)
                if len(vals_bytes) < count * 2:
                    break
                offset += count * 2

                frame_end = offset

                idx.append({
                    "frame_start": frame_start,
                    "head40_offset": frame_start,
                    "mask_offset": mask_offset,
                    "vals_offset": vals_offset,
                    "frame_end": frame_end,
                    "ny": ny,
                    "nx": nx,
                    "thr": thr,
                    "count": count,
                    "mbytes": mbytes,
                })

        self._index = idx
        self._nframes = len(idx)

    @property
    def nframes(self):
        self._build_index()
        return self._nframes

    # ------------------------------------------------------------------
    # low-level readers
    # ------------------------------------------------------------------
    def _read_head40(self, i: int):
        self._build_index()
        rec = self._index[i]

        with open(self.fname, "rb") as f:
            f.seek(rec["head40_offset"])
            head40 = f.read(40)
            if len(head40) < 40:
                raise RuntimeError(f"failed to read head40 for frame {i}")
        return head40

    def _read_orig32_u32(self, i: int):
        head40 = self._read_head40(i)
        orig32 = head40[8:40]
        return np.frombuffer(orig32, dtype="<u4", count=8)

    # ------------------------------------------------------------------
    # public head access
    # ------------------------------------------------------------------
    def get_head(self, i: int):
        """
        Return combined 40B head as 10 uint32:
          word0, word1   : SSI
          word2 ... word9: original 32B header words
        """
        head40 = self._read_head40(i)
        return np.frombuffer(head40, dtype="<u4", count=10)

    def summary_head(self, i: int = 0):
        h = self.get_head(i)
        for k, v in enumerate(h):
            print(f"word{k} = {v}")

    def get_word(self, j: int):
        self._build_index()
        out = np.empty(self._nframes, dtype=np.uint32)
        for i in range(self._nframes):
            out[i] = self.get_head(i)[j]
        return out

    # ------------------------------------------------------------------
    # timestamp helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _decode_time_from_head40(head40: bytes):
        if len(head40) < 40:
            raise RuntimeError("head40 too short")

        usec = struct.unpack_from("<I", head40, 28)[0]
        sec  = struct.unpack_from("<I", head40, 36)[0]
        dt = datetime.fromtimestamp(sec) + timedelta(microseconds=usec)
        return dt, sec, usec

    def get_datetime(self, i: int):
        head40 = self._read_head40(i)
        return self._decode_time_from_head40(head40)[0]

    def read_first_time(self):
        """
        Fast path: read only the first frame header from disk.
        No need to build the full frame index.
        """
        with open(self.fname, "rb") as f:
            head40 = f.read(40)
            if len(head40) < 40:
                raise RuntimeError("file too short to contain first L1BM frame header")

        return self._decode_time_from_head40(head40)[0]

    def read_last_time(self):
        self._build_index()
        if self._nframes == 0:
            raise RuntimeError("no complete L1BM frame found")
        return self.get_datetime(self._nframes - 1)

    def read_time_range(self):
        return (self.read_first_time(), self.read_last_time())

    # ------------------------------------------------------------------
    # frame metadata
    # ------------------------------------------------------------------
    def get_shape(self, i: int):
        self._build_index()
        rec = self._index[i]
        return rec["ny"], rec["nx"]

    def get_thr(self, i: int):
        self._build_index()
        return self._index[i]["thr"]

    def get_count(self, i: int):
        self._build_index()
        return self._index[i]["count"]

    # ------------------------------------------------------------------
    # payload readers
    # ------------------------------------------------------------------
    def get_mask(self, i: int):
        self._build_index()
        rec = self._index[i]

        with open(self.fname, "rb") as f:
            f.seek(rec["mask_offset"])
            mask_bytes = f.read(rec["mbytes"])
            if len(mask_bytes) < rec["mbytes"]:
                raise RuntimeError(f"failed to read mask bytes for frame {i}")

        mask = np.unpackbits(np.frombuffer(mask_bytes, np.uint8), bitorder="big")
        mask = mask[: rec["ny"] * rec["nx"]].reshape(rec["ny"], rec["nx"]).astype(bool)
        return mask

    def get_vals(self, i: int):
        self._build_index()
        rec = self._index[i]

        with open(self.fname, "rb") as f:
            f.seek(rec["vals_offset"])
            vals_bytes = f.read(rec["count"] * 2)
            if len(vals_bytes) < rec["count"] * 2:
                raise RuntimeError(f"failed to read vals for frame {i}")

        vals = np.frombuffer(vals_bytes, dtype="<u2", count=rec["count"])
        return vals

    def get_frame(self, i: int):
        """
        Return a dict similar to your iter_l1bm_frames output.
        """
        self._build_index()
        rec = self._index[i]
        head40 = self._read_head40(i)

        ssi = head40[:8]
        orig_u32 = np.frombuffer(head40[8:40], dtype="<u4", count=8)
        mask = self.get_mask(i)
        vals = self.get_vals(i)

        return {
            "ssi": ssi,
            "orig_u32": orig_u32,
            "ny": rec["ny"],
            "nx": rec["nx"],
            "thr": rec["thr"],
            "mask": mask,
            "vals": vals,
            "count": rec["count"],
            "mbytes": rec["mbytes"],
        }
        
    def reconstruct_frame(self, i, fill_value=0, dtype=np.uint16):
        """
        Reconstruct one full 2D frame from L1BM compressed representation.

        Parameters
        ----------
        i : int
            Frame index.
        fill_value : scalar
            Value for pixels not included in the mask.
        dtype : numpy dtype
            Output dtype.

        Returns
        -------
        img : np.ndarray
            Reconstructed image with shape (ny, nx).
        """
        self._build_index()
        rec = self._index[i]

        mask = self.get_mask(i)
        vals = self.get_vals(i)

        img = np.full((rec["ny"], rec["nx"]), fill_value, dtype=dtype)

        flat = img.ravel()
        hit_idx = np.flatnonzero(mask.ravel())

        if hit_idx.size != vals.size:
            raise RuntimeError(
                f"mask/value mismatch in frame {i}: mask hits={hit_idx.size}, vals={vals.size}"
            )

        flat[hit_idx] = vals.astype(dtype, copy=False)
        return img