import torch
import numpy as np
from copy import copy
from pathlib import Path
import pytorch_lightning as pl
from argparse import ArgumentParser
from data_utils.dataset import CMambaDataset, DataConverter


def worker_init_fn(worker_id):
    """
    Handle random seeding.
    """
    worker_info = torch.utils.data.get_worker_info()
    data = worker_info.dataset  # pylint: disable=no-member

    # Check if we are using DDP
    is_ddp = False
    if torch.distributed.is_available():
        if torch.distributed.is_initialized():
            is_ddp = True

    # for NumPy random seed we need it to be in this range
    base_seed = worker_info.seed  # pylint: disable=no-member

    if is_ddp:  # DDP training: unique seed is determined by worker and device
        seed = base_seed + torch.distributed.get_rank() * worker_info.num_workers
    else:
        seed = base_seed


class CMambaDataModule(pl.LightningDataModule):

    def __init__(
        self,
        data_config,
        train_transform,
        val_transform,
        test_transform,
        batch_size,
        distributed_sampler,
        num_workers=4,
        normalize=False,
        window_size=14,
    ):

        super().__init__()

        self.data_config = data_config
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed_sampler = distributed_sampler
        self.window_size = window_size
        self.factors = None

        self.converter = DataConverter(data_config)
        train, val, test = self.converter.get_data()
        self.data_dict = {
            "train": train,
            "val": val,
            "test": test,
        }

        if normalize:
            self.normalize()

    def normalize(self):
        train_data = self.data_dict.get("train")
        if train_data is None:
            print("Warning: Training data not found. Skipping normalization.")
            self.factors = None
            return

        factors = {}
        # --- 수정: Training set 만으로 min/max 계산 ---
        print("Calculating normalization factors using TRAINING data only...")
        for key in train_data.keys():
            if key in train_data and len(train_data[key]) > 0:
                try:
                    # numpy 배열로 변환하여 효율적으로 계산하고, 숫자형 데이터인지 확인
                    feature_data = np.asarray(train_data[key])
                    if np.issubdtype(feature_data.dtype, np.number):
                        min_val = np.min(feature_data)
                        max_val = np.max(feature_data)
                        scale = max_val - min_val
                        shift = min_val

                        # min == max 인 경우 (상수 값 피처) 처리: scale=1, shift=min_val -> 정규화 결과는 0
                        if abs(scale) < 1e-9:  # 부동소수점 오차 감안
                            print(
                                f"Warning: Feature '{key}' has a constant value ({min_val}) in the training set. Normalizing to 0."
                            )
                            factors[key] = {
                                "min": min_val,
                                "max": max_val,
                                "scale": 1.0,
                                "shift": min_val,
                            }
                        else:
                            factors[key] = {
                                "min": min_val,
                                "max": max_val,
                                "scale": scale,
                                "shift": shift,
                            }
                    else:
                        print(
                            f"Warning: Feature '{key}' is not numeric. Skipping normalization for this feature."
                        )
                        factors[key] = None  # 이 피처는 정규화 건너뜀
                except Exception as e:
                    print(
                        f"Warning: Could not process feature '{key}'. Error: {e}. Skipping normalization."
                    )
                    factors[key] = None  # 에러 발생 시 건너뜀
            else:
                print(
                    f"Warning: Feature '{key}' not found or empty in training data. Skipping normalization."
                )
                factors[key] = None  # 데이터 없으면 건너뜀

        self.factors = factors  # 계산된 factors 저장 (train 기준)

        # --- 수정: 모든 데이터 분할에 Training set 기준 factors 적용 ---
        print("Applying normalization to all data splits...")
        for split, data in self.data_dict.items():
            if data is None:
                continue
            print(f"Normalizing split: {split}")
            for key in data.keys():
                # 해당 key에 대한 factors가 정상적으로 계산되었는지 확인
                if key in self.factors and self.factors[key] is not None:
                    factor_info = self.factors[key]
                    scale = factor_info.get("scale")
                    shift = factor_info.get("shift")

                    # 원본 타임스탬프 백업 (정규화 성공 여부와 관계없이)
                    if key == "Timestamp":
                        data["Timestamp_orig"] = copy(data.get(key))

                    # scale 값이 유효한 경우에만 정규화 적용
                    if scale is not None and shift is not None and abs(scale) > 1e-9:
                        try:
                            feature_data = np.asarray(data[key])
                            if np.issubdtype(feature_data.dtype, np.number):
                                data[key] = (feature_data - shift) / scale
                            # else: 숫자형 아니면 그대로 둠
                        except Exception as e:
                            print(
                                f"Warning: Could not normalize feature '{key}' in split '{split}'. Error: {e}"
                            )
                    elif abs(scale) < 1e-9:  # scale이 0에 가까운 경우 (상수값 처리)
                        try:
                            feature_data = np.asarray(data[key])
                            if np.issubdtype(feature_data.dtype, np.number):
                                # factors 딕셔너리 생성 시 scale=1, shift=min_val로 설정했으므로 아래 계산은 0이 됨
                                data[key] = (feature_data - shift) / scale
                        except Exception as e:
                            print(
                                f"Warning: Could not normalize constant feature '{key}' in split '{split}'. Error: {e}"
                            )

    def _create_data_loader(self, data_split, data_transform, batch_size=None):
        dataset = CMambaDataset(
            data=self.data_dict.get(data_split),
            split=data_split,
            window_size=self.window_size,
            transform=data_transform,
        )

        batch_size = self.batch_size if batch_size is None else batch_size
        sampler = (
            torch.utils.data.DistributedSampler(dataset)
            if self.distributed_sampler
            else None
        )

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
            sampler=sampler,
            drop_last=False,
        )
        return dataloader

    def train_dataloader(self):
        return self._create_data_loader(
            data_split="train", data_transform=self.train_transform
        )

    def val_dataloader(self):
        return self._create_data_loader(
            data_split="val", data_transform=self.val_transform
        )

    def test_dataloader(self):
        return self._create_data_loader(
            data_split="test", data_transform=self.test_transform
        )
