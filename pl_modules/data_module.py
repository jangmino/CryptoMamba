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
        train_df, val_df, test_df = self.converter.get_data()

        # [수정] DataFrame을 NumPy 배열 딕셔너리로 변환
        self.data_dict = {
            "train": self._df_to_numpy_dict(train_df),
            "val": self._df_to_numpy_dict(val_df),
            "test": self._df_to_numpy_dict(test_df),
        }

        if normalize:
            self.normalize()

    # [추가] DataFrame을 NumPy 딕셔너리로 변환하는 헬퍼 함수
    def _df_to_numpy_dict(self, df):
        if df is None:
            return None
        return {col: df[col].to_numpy() for col in df.columns}

    def normalize(self):
        train_data = self.data_dict.get("train")
        if train_data is None:
            print("Warning: Training data not found. Skipping normalization.")
            self.factors = None
            return

        factors = {}
        print("Calculating normalization factors using TRAINING data only...")
        for key, feature_data in train_data.items():
            if np.issubdtype(feature_data.dtype, np.number):
                min_val = np.min(feature_data)
                max_val = np.max(feature_data)
                scale = max_val - min_val
                shift = min_val
                if abs(scale) < 1e-9:
                    print(
                        f"Warning: Feature '{key}' has a constant value. Normalizing to 0."
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
                factors[key] = None

        self.factors = factors

        print("Applying normalization to all data splits...")
        for split, data in self.data_dict.items():
            if data is None:
                continue
            print(f"Normalizing split: {split}")

            # 원본 타임스탬프 백업
            if "Timestamp" in data:
                data["Timestamp_orig"] = data["Timestamp"].copy()

            for key, feature_data in data.items():
                if key in self.factors and self.factors[key] is not None:
                    factor_info = self.factors[key]
                    scale = factor_info.get("scale")
                    shift = factor_info.get("shift")
                    if scale is not None and shift is not None and abs(scale) > 1e-9:
                        data[key] = (feature_data - shift) / scale
                    elif abs(scale) < 1e-9:  # 상수 값 처리
                        data[key] = (feature_data - shift) / scale

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
