import os
import struct
import numpy as np
from datetime import datetime, timedelta, timezone


class L1BMDriver:
    E1_FMT = "<4sHHHHII"
    E1_SZ = struct.calcsize(E1_FMT)
    HEAD40_BYTES = 40

    def __init__(self, fname: str):
        self.fname = fname
        self._index = None
        self._nframes = None

    @staticmethod
    def reconstruct(ny, nx, mask, vals, fill=0, dtype=np.uint16):
        if mask.shape != (ny, nx):
            raise ValueError(f"mask shape mismatch: got {mask.shape}, expected {(ny, nx)}")
        nhit = int(mask.sum())
        if nhit != len(vals):
            raise RuntimeError(f"mask/value mismatch: mask hits={nhit}, vals={len(vals)}")
        img = np.full((ny, nx), fill, dtype=dtype)
        img.ravel()[mask.ravel()] = vals.astype(dtype, copy=False)
        return img

    @staticmethod
    def _decode_time_from_head40(head40: bytes):
        if len(head40) < 40:
            raise RuntimeError("head40 too short")
        usec = struct.unpack_from("<I", head40, 28)[0]
        sec = struct.unpack_from("<I", head40, 36)[0]
        dt = datetime.fromtimestamp(sec, tz=timezone.utc) + timedelta(microseconds=usec)
        return dt, sec, usec

    def _build_index(self):
        if self._index is not None:
            return

        idx = []
        offset = 0
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
                f.seek(mbytes, os.SEEK_CUR)
                offset += mbytes

                vals_offset = offset
                f.seek(count * 2, os.SEEK_CUR)
                offset += count * 2

                idx.append({
                    "frame_start": frame_start,
                    "head40_offset": frame_start,
                    "mask_offset": mask_offset,
                    "vals_offset": vals_offset,
                    "frame_end": offset,
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

    @property
    def index(self):
        self._build_index()
        return self._index

    def _read_head40(self, i: int):
        self._build_index()
        rec = self._index[i]
        with open(self.fname, "rb") as f:
            f.seek(rec["head40_offset"])
            head40 = f.read(self.HEAD40_BYTES)
        if len(head40) < self.HEAD40_BYTES:
            raise RuntimeError(f"failed to read head40 for frame {i}")
        return head40

    def _read_mask_vals(self, i: int):
        self._build_index()
        rec = self._index[i]
        with open(self.fname, "rb") as f:
            f.seek(rec["mask_offset"])
            mask_bytes = f.read(rec["mbytes"])
            if len(mask_bytes) < rec["mbytes"]:
                raise RuntimeError(f"failed to read mask bytes for frame {i}")
            vals_bytes = f.read(rec["count"] * 2)
            if len(vals_bytes) < rec["count"] * 2:
                raise RuntimeError(f"failed to read vals for frame {i}")

        mask = np.unpackbits(np.frombuffer(mask_bytes, dtype=np.uint8), bitorder="big")
        mask = mask[:rec["ny"] * rec["nx"]].reshape(rec["ny"], rec["nx"]).astype(bool, copy=False)
        vals = np.frombuffer(vals_bytes, dtype="<u2", count=rec["count"])
        return mask, vals

    def get_head(self, i: int):
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

    def get_datetime(self, i: int):
        return self._decode_time_from_head40(self._read_head40(i))[0]

    def read_first_time(self):
        with open(self.fname, "rb") as f:
            head40 = f.read(self.HEAD40_BYTES)
        if len(head40) < self.HEAD40_BYTES:
            raise RuntimeError("file too short to contain first L1BM frame header")
        return self._decode_time_from_head40(head40)[0]

    def read_last_time(self):
        self._build_index()
        if self._nframes == 0:
            raise RuntimeError("no complete L1BM frame found")
        return self.get_datetime(self._nframes - 1)

    def read_time_range(self):
        return self.read_first_time(), self.read_last_time()

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

    def get_mask(self, i: int):
        return self._read_mask_vals(i)[0]

    def get_vals(self, i: int):
        return self._read_mask_vals(i)[1]

    def get_frame(self, i: int, reconstruct_image=False, fill_value=0, dtype=np.uint16):
        self._build_index()
        rec = self._index[i]
        head40 = self._read_head40(i)
        mask, vals = self._read_mask_vals(i)

        out = {
            "ssi": head40[:8],
            "orig_u32": np.frombuffer(head40[8:40], dtype="<u4", count=8),
            "ny": rec["ny"],
            "nx": rec["nx"],
            "thr": rec["thr"],
            "mask": mask,
            "vals": vals,
            "count": rec["count"],
            "mbytes": rec["mbytes"],
        }
        if reconstruct_image:
            out["img"] = self.reconstruct(rec["ny"], rec["nx"], mask, vals, fill=fill_value, dtype=dtype)
        return out

    def reconstruct_frame(self, i, fill_value=0, dtype=np.uint16):
        fr = self.get_frame(i)
        return self.reconstruct(fr["ny"], fr["nx"], fr["mask"], fr["vals"], fill=fill_value, dtype=dtype)

    def read_one_frame_from_file(self, f, reconstruct_image=False, fill_value=0, dtype=np.uint16):
        ssi = f.read(8)
        if not ssi:
            return None
        if len(ssi) < 8:
            raise RuntimeError("truncated frame: SSI < 8 bytes")

        orig32 = f.read(32)
        if len(orig32) < 32:
            raise RuntimeError("truncated frame: orig32 < 32 bytes")
        orig_u32 = np.frombuffer(orig32, dtype="<u4", count=8)

        e1hdr = f.read(self.E1_SZ)
        if len(e1hdr) < self.E1_SZ:
            raise RuntimeError("truncated frame: E1 header incomplete")

        magic, ny, nx, thr, _rsv, count, mbytes = struct.unpack(self.E1_FMT, e1hdr)
        if magic != b"E1BM":
            raise RuntimeError(f"bad frame magic: {magic!r}, expected b'E1BM'")

        gap4 = f.read(4)
        if len(gap4) < 4:
            raise RuntimeError("truncated frame: missing 4-byte gap")

        mask_bytes = f.read(mbytes)
        if len(mask_bytes) < mbytes:
            raise RuntimeError(f"truncated frame: mask_bytes={len(mask_bytes)}, expected={mbytes}")

        vals_bytes = f.read(count * 2)
        if len(vals_bytes) < count * 2:
            raise RuntimeError(f"truncated frame: vals_bytes={len(vals_bytes)}, expected={count * 2}")

        mask = np.unpackbits(np.frombuffer(mask_bytes, dtype=np.uint8), bitorder="big")
        mask = mask[:ny * nx].reshape(ny, nx).astype(bool, copy=False)
        vals = np.frombuffer(vals_bytes, dtype="<u2", count=count)

        out = {
            "ssi": ssi,
            "orig_u32": orig_u32,
            "ny": ny,
            "nx": nx,
            "thr": thr,
            "mask": mask,
            "vals": vals,
            "count": count,
            "mbytes": mbytes,
        }
        if reconstruct_image:
            out["img"] = self.reconstruct(ny, nx, mask, vals, fill=fill_value, dtype=dtype)
        return out

    def read_first(self, reconstruct_image=True, fill_value=0, dtype=np.uint16):
        with open(self.fname, "rb") as f:
            frame = self.read_one_frame_from_file(f, reconstruct_image=reconstruct_image, fill_value=fill_value, dtype=dtype)
        if frame is None:
            raise RuntimeError("empty file or no complete L1BM frame")
        return frame

    def iter_frames(self, reconstruct_image=False, fill_value=0, dtype=np.uint16):
        with open(self.fname, "rb") as f:
            while True:
                frame = self.read_one_frame_from_file(f, reconstruct_image=reconstruct_image, fill_value=fill_value, dtype=dtype)
                if frame is None:
                    break
                yield frame