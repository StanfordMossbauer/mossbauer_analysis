import struct
from datetime import datetime, timedelta
import numpy as np 

HEAD40_BYTES = 40

# A general function that does not need to read the whole frame to get the first timestamp; 
# A more precise way;
def read_first_frame_time(fname):
    """
    word5 @ offset 28 = microseconds
    word7 @ offset 36 = seconds
    Returns the utc timestamp; 
    """
    with open(fname, "rb") as f:
        head40 = f.read(HEAD40_BYTES)
        if len(head40) < HEAD40_BYTES:
            raise RuntimeError("file too short to contain first frame header")

    usec = struct.unpack_from("<I", head40, 28)[0]   # 8 + 20
    sec  = struct.unpack_from("<I", head40, 36)[0]   # 8 + 28

    return datetime.fromtimestamp(sec,tz=timezone.utc) + timedelta(microseconds=usec)

# Convert the image
def convert_image(image: np.ndarray) -> np.ndarray:
    row, col4 = image.shape
    col = col4 // 4
    im1 = image[:, 0*col : 1*col]
    im2 = image[:, 1*col : 2*col]
    im3 = image[::-1, 2*col : 3*col]
    im4 = image[::-1, 3*col : 4*col]
    return np.vstack([np.hstack([im3, im4]),
                        np.hstack([im1, im2])])