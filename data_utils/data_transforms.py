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
        클래스 인스턴스를 함수처럼 호출할 때 실행되는 메서드입니다.
        주어진 데이터 창(window)을 모델 입력 형식으로 변환합니다.

        Args:
            window (pd.DataFrame or similar): 시계열 데이터의 한 부분을 나타내는 데이터 창입니다.

        Returns:
            dict: 모델 학습/추론에 필요한 데이터를 담은 딕셔너리.
                  'features': 모델 입력 특성 텐서 (특성 수, 창 길이 - 1)
                  '<key>': 각 키에 해당하는 마지막 시점의 값
                  '<key>_old': 각 키에 해당하는 마지막에서 두 번째 시점의 값
                  'Timestamp_orig': (존재할 경우) 정규화 전 원본 타임스탬프의 마지막 값
        """
        data_list = []
        output = {}
        if "Timestamp_orig" in window.keys():
            self.keys.append("Timestamp_orig")
        for key in self.keys:
            data = torch.tensor(window.get(key).tolist())
            if key == "Volume":
                data /= 1e9
            output[key] = data[-1]
            output[f"{key}_old"] = data[-2]
            if key == "Timestamp_orig":
                continue
            data_list.append(data[:-1].reshape(1, -1))
        features = torch.cat(data_list, 0)
        output["features"] = features
        # raise ValueError(output)
        return output

    def set_initial_seed(self, seed):
        self.rng.seed(seed)
