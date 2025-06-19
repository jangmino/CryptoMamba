import torch
import numpy as np


class DataTransform:
    def __init__(
        self, is_train, use_volume=False, additional_features=[], output_type="torch"
    ):
        """
        output_type: "torch" or "numpy"
        """
        self.is_train = is_train
        self.keys = ["Timestamp", "Open", "High", "Low", "Close"]
        if use_volume:
            self.keys.append("Volume")
        self.keys += additional_features
        self.output_type = output_type
        print(f"DataTransform keys: {self.keys}, output_type: {self.output_type}")

    def __call__(self, window):
        data_list = []
        output = {}

        keys_in_window = list(window.keys())

        for key in self.keys:
            if key not in keys_in_window:
                continue

            arr = window.get(key)
            if self.output_type == "torch":
                data = torch.tensor(arr, dtype=torch.float32)
            else:
                data = np.asarray(arr, dtype=np.float32)

            if key == "Volume":
                data = data / 1e9
            output[key] = data[-1]
            output[f"{key}_old"] = data[-2]
            if "Timestamp_orig" in key:
                continue
            data_list.append(data[:-1].reshape(1, -1))

        if self.output_type == "torch":
            features = torch.cat(data_list, 0)
        else:
            features = np.concatenate(data_list, 0)
        output["features"] = features
        return output

    def set_initial_seed(self, seed):
        if hasattr(self, "rng"):
            self.rng.seed(seed)
