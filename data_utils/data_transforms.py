import torch


class DataTransform:
    def __init__(self, is_train, use_volume=False, additional_features=[]):
        self.is_train = is_train
        self.keys = ["Timestamp", "Open", "High", "Low", "Close"]
        if use_volume:
            self.keys.append("Volume")
        self.keys += additional_features
        print(self.keys)

    def __call__(self, window):
        """
        [수정] 이제 window는 DataFrame이 아닌 NumPy 배열의 딕셔너리입니다.
        """
        data_list = []
        output = {}

        # [수정] window가 딕셔너리이므로 keys()를 바로 사용
        keys_in_window = list(window.keys())
        # Timestamp_orig가 있으면 키 리스트에 추가 (data_module에서 처리)
        # if "Timestamp_orig" not in self.keys and "Timestamp_orig" in keys_in_window:
        #     self.keys.append("Timestamp_orig")

        for key in self.keys:
            if key not in keys_in_window:
                continue

            # [수정] .get(key).tolist() 대신 바로 텐서로 변환
            data = torch.tensor(window.get(key), dtype=torch.float32)

            if key == "Volume":
                data /= 1e9
            output[key] = data[-1]
            output[f"{key}_old"] = data[-2]
            if "Timestamp_orig" in key:  # Timestamp_orig는 피처에 포함하지 않음
                continue
            data_list.append(data[:-1].reshape(1, -1))

        features = torch.cat(data_list, 0)
        output["features"] = features
        return output

    def set_initial_seed(self, seed):
        self.rng.seed(seed)
