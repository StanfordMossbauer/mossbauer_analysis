import numpy as np


class doubleblock:
    def __init__(self,
                 mode,
                 single_data=None,
                 single_head=None,
                 forward_data=None,
                 forward_head=None,
                 backward_data=None,
                 backward_head=None,
                 start=3,
                 end=45,
                 up=3,
                 down=42,
                 filter_mask=None):
        self.mode = mode
        self.single_data = single_data
        self.single_head = single_head
        self.forward_data = forward_data
        self.forward_head = forward_head
        self.backward_data = backward_data
        self.backward_head = backward_head

        self.start = start
        self.end = end
        self.up = up
        self.down = down

        self.filter_mask = filter_mask
        self.ds_flat = self.build_ds_flat()

    @classmethod
    def from_reader(cls,
                    double_reader,
                    start=3,
                    end=45,
                    up=3,
                    down=42,
                    filter_mask=None):
        if double_reader.mode == "single":
            return cls(
                mode="single",
                single_data=double_reader.single_reader.data,
                single_head=double_reader.single_reader.head,
                start=start,
                end=end,
                up=up,
                down=down,
                filter_mask=filter_mask,
            )

        if double_reader.mode == "double":
            return cls(
                mode="double",
                forward_data=double_reader.forward_reader.data,
                forward_head=double_reader.forward_reader.head,
                backward_data=double_reader.backward_reader.data,
                backward_head=double_reader.backward_reader.head,
                start=start,
                end=end,
                up=up,
                down=down,
                filter_mask=filter_mask,
            )

        raise ValueError("double_reader mode is not valid.")

    def build_base_mask(self):
        ds = np.zeros((4, 44, 192), dtype=np.float32)
        for i in range(4):
            ds[i,
               self.up:self.down,
               self.start + 48 * i:self.end + 48 * i] = 1.0
        return ds

    def build_ds_flat(self):
        ds = self.build_base_mask()

        if self.filter_mask is not None:
            filt = np.asarray(self.filter_mask, dtype=np.float32)

            if filt.shape == (44, 192):
                ds = ds * filt[None, :, :]
            elif filt.shape == (4, 44, 192):
                ds = ds * filt
            else:
                raise ValueError(
                    f"filter_mask shape must be (44,192) or (4,44,192), got {filt.shape}"
                )

        return ds.reshape(4, -1).T

    def set_filter(self, filter_mask):
        self.filter_mask = filter_mask
        self.ds_flat = self.build_ds_flat()

    @staticmethod
    def count_one_block(data, head, ds_flat):
        if data is None or head is None:
            return np.zeros((2, 4), dtype=np.float64)

        t_counts = data.reshape(data.shape[0], -1) @ ds_flat
        word6 = head[:, 6]

        out = np.zeros((2, 4), dtype=np.float64)
        out[0] = t_counts[word6 == 0].sum(axis=0)
        out[1] = t_counts[word6 == 1].sum(axis=0)
        return out

    def get16(self):
        out = np.zeros((2, 2, 4), dtype=np.float64)

        if self.mode == "single":
            out[0] = self.count_one_block(
                self.single_data,
                self.single_head,
                self.ds_flat
            )
            return out.reshape(-1)

        if self.mode == "double":
            out[0] = self.count_one_block(
                self.forward_data,
                self.forward_head,
                self.ds_flat
            )
            out[1] = self.count_one_block(
                self.backward_data,
                self.backward_head,
                self.ds_flat
            )
            return out.reshape(-1)

        raise ValueError("Invalid mode.")