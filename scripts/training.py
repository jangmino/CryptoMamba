import os, sys, pathlib
import numpy as np
import xgboost as xgb

sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import yaml
from utils import io_tools
import pytorch_lightning as pl
from argparse import ArgumentParser
from pl_modules.data_module import CMambaDataModule
from data_utils.data_transforms import DataTransform
from data_utils.dataset import CMambaDataset
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


ROOT = io_tools.get_root(__file__, num_returns=2)


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--logdir",
        type=str,
        help="Logging directory.",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="gpu",
        help="The type of accelerator.",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of computing devices.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=23,
        help="Logging directory.",
    )
    parser.add_argument(
        "--expname",
        type=str,
        default="Cmamba",
        help="Experiment name. Reconstructions will be saved under this folder.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="cmamba_nv",
        help="Path to config file.",
    )
    parser.add_argument(
        "--logger_type",
        default="tb",
        type=str,
        help="Path to config file.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel workers.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="batch_size",
    )
    parser.add_argument(
        "--save_checkpoints",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--use_volume",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--use_xgboost",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
    )

    parser.add_argument(
        "--max_epochs",
        type=int,
        default=200,
    )

    args = parser.parse_args()
    return args


def save_all_hparams(log_dir, args):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    save_dict = vars(args)
    path = log_dir + "/hparams.yaml"
    if os.path.exists(path):
        return
    with open(path, "w") as f:
        yaml.dump(save_dict, f)


def load_model(config, logger_type):
    arch_config = io_tools.load_config_from_yaml("configs/models/archs.yaml")
    model_arch = config.get("model")
    model_config_path = f"{ROOT}/configs/models/{arch_config.get(model_arch)}"
    model_config = io_tools.load_config_from_yaml(model_config_path)

    normalize = model_config.get("normalize", False)
    hyperparams = config.get("hyperparams")
    if hyperparams is not None:
        for key in hyperparams.keys():
            model_config.get("params")[key] = hyperparams.get(key)

    model_config.get("params")["logger_type"] = logger_type
    model = io_tools.instantiate_from_config(model_config)
    model.cuda()
    model.train()
    return model, normalize


def load_xgboost_config(config, logger_type):
    arch_config = io_tools.load_config_from_yaml("configs/models/archs.yaml")
    model_arch = config.get("model")
    model_config_path = f"{ROOT}/configs/models/{arch_config.get(model_arch)}"
    model_config = io_tools.load_config_from_yaml(model_config_path)

    normalize = model_config.get("normalize", False)
    hyperparams = config.get("hyperparams")
    if hyperparams is not None:
        for key in hyperparams.keys():
            model_config.get("params")[key] = hyperparams.get(key)

    model_config.get("params")["logger_type"] = logger_type

    return model_config


def train_lightning(args):
    pl.seed_everything(args.seed)
    logdir = args.logdir

    config = io_tools.load_config_from_yaml(
        f"{ROOT}/configs/training/{args.config}.yaml"
    )

    data_config = io_tools.load_config_from_yaml(
        f"{ROOT}/configs/data_configs/{config.get('data_config')}.yaml"
    )
    use_volume = args.use_volume

    if not use_volume:
        use_volume = config.get("use_volume")
    train_transform = DataTransform(
        is_train=True,
        use_volume=use_volume,
        additional_features=data_config.get("additional_features", []),
    )
    val_transform = DataTransform(
        is_train=False,
        use_volume=use_volume,
        additional_features=data_config.get("additional_features", []),
    )
    test_transform = DataTransform(
        is_train=False,
        use_volume=use_volume,
        additional_features=data_config.get("additional_features", []),
    )

    model, normalize = load_model(config, args.logger_type)

    # name = config.get("name", args.expname)
    if args.logger_type == "tb":
        logger = TensorBoardLogger("logs", name=args.expname)
        logger.log_hyperparams(args)
    elif args.logger_type == "wandb":
        tmp = vars(args)
        tmp.update(config)
        logger = pl.loggers.WandbLogger(project=args.expname, config=tmp)
    elif args.logger_type == "none":
        logger = False
    else:
        raise ValueError("Unknown logger type.")

    data_module = CMambaDataModule(
        data_config,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        batch_size=args.batch_size,
        distributed_sampler=True,
        num_workers=args.num_workers,
        normalize=normalize,
        window_size=model.window_size,
    )

    callbacks = []
    if args.save_checkpoints:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            save_top_k=1,
            verbose=True,
            monitor="val/rmse",
            mode="min",
            filename="epoch{epoch}-val-rmse{val/rmse:.4f}",
            auto_insert_metric_name=False,
            save_last=True,
        )
        callbacks.append(checkpoint_callback)

    max_epochs = config.get("max_epochs", args.max_epochs)
    model.set_normalization_coeffs(data_module.factors)

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=max_epochs,
        enable_checkpointing=args.save_checkpoints,
        log_every_n_steps=10,
        logger=logger,
        callbacks=callbacks,
        strategy=DDPStrategy(find_unused_parameters=False),
    )

    trainer.fit(model, datamodule=data_module)
    if args.save_checkpoints:
        trainer.test(
            model, datamodule=data_module, ckpt_path=checkpoint_callback.best_model_path
        )


def build_xgboost_dataset(ds, y_key="Close"):
    X = []
    y = []
    y_old = []
    for i in range(len(ds)):
        sample = ds[i]
        if not isinstance(sample, dict):
            raise ValueError(f"Expected sample to be a dict, got {type(sample)}")
        if "features" not in sample or y_key not in sample:
            raise KeyError(f"Sample must contain 'features' and '{y_key}' keys.")
        if not isinstance(sample["features"], np.ndarray):
            raise TypeError(
                f"Expected 'features' to be a numpy array, got {type(sample['features'])}"
            )
        features = sample["features"]
        target = sample[y_key]
        target_old = sample[f"{y_key}_old"]
        X.append(features)
        y.append(target)
        y_old.append(target_old)
    np_X = np.array(X)
    np_X = np_X.reshape(np_X.shape[0], -1)  # Flatten features if needed
    return np_X, np.array(y), np.array(y_old)


def denormalize_xgboost_predictions(y, y_hat, factors, y_key="Close"):
    if factors is None:
        return y, y_hat
    scale = factors.get(y_key).get("max") - factors.get(y_key).get("min")
    shift = factors.get(y_key).get("min")
    y = y * scale + shift
    y_hat = y_hat * scale + shift
    return y, y_hat


#############################
def train_xgboost(args):
    pl.seed_everything(args.seed)
    logdir = args.logdir

    config = io_tools.load_config_from_yaml(
        f"{ROOT}/configs/training/{args.config}.yaml"
    )

    data_config = io_tools.load_config_from_yaml(
        f"{ROOT}/configs/data_configs/{config.get('data_config')}.yaml"
    )
    use_volume = args.use_volume

    if not use_volume:
        use_volume = config.get("use_volume")
    train_transform = DataTransform(
        is_train=True,
        use_volume=use_volume,
        additional_features=data_config.get("additional_features", []),
        output_type="numpy",
    )
    val_transform = DataTransform(
        is_train=False,
        use_volume=use_volume,
        additional_features=data_config.get("additional_features", []),
        output_type="numpy",
    )
    test_transform = DataTransform(
        is_train=False,
        use_volume=use_volume,
        additional_features=data_config.get("additional_features", []),
        output_type="numpy",
    )

    xgboost_config = load_xgboost_config(config, args.logger_type)

    # name = config.get("name", args.expname)
    if args.logger_type == "tb":
        logger = TensorBoardLogger("logs", name=args.expname)
        logger.log_hyperparams(args)
    elif args.logger_type == "wandb":
        tmp = vars(args)
        tmp.update(config)
        logger = pl.loggers.WandbLogger(project=args.expname, config=tmp)
    elif args.logger_type == "none":
        logger = False
    else:
        raise ValueError("Unknown logger type.")

    data_module = CMambaDataModule(
        data_config,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        batch_size=args.batch_size,
        distributed_sampler=False,  # XGBoost does not use distributed sampler
        num_workers=args.num_workers,
        normalize=xgboost_config.get("normalize", False),
        window_size=xgboost_config.get("params").get("window_size", 14),
    )

    train_ds = CMambaDataset(
        data_module.data_dict.get("train"),
        split="train",
        window_size=data_module.window_size,
        transform=train_transform,
    )
    val_ds = CMambaDataset(
        data_module.data_dict.get("val"),
        split="val",
        window_size=data_module.window_size,
        transform=val_transform,
    )
    test_ds = CMambaDataset(
        data_module.data_dict.get("test"),
        split="test",
        window_size=data_module.window_size,
        transform=test_transform,
    )

    y_key = xgboost_config.get("params").get("y_key", "Close")
    train_X, train_Y, train_Y_old = build_xgboost_dataset(train_ds, y_key=y_key)
    val_X, val_Y, val_Y_old = build_xgboost_dataset(val_ds, y_key=y_key)
    test_X, test_Y, test_Y_old = build_xgboost_dataset(test_ds, y_key=y_key)

    if xgboost_config.get("mode", "default") == "diff":
        target_train_Y = train_Y - train_Y_old
        target_val_Y = val_Y - val_Y_old
        target_test_Y = test_Y - test_Y_old
    else:
        target_train_Y = train_Y
        target_val_Y = val_Y
        target_test_Y = test_Y

    model = xgb.XGBRegressor(**xgboost_config.get("params")["xgb_params"])
    model.fit(train_X, target_train_Y, eval_set=[(val_X, target_val_Y)], verbose=True)

    # Evaluate on training set
    print("Evaluating on training set...")
    y_train_pred = model.predict(train_X)
    if xgboost_config.get("mode", "default") == "diff":
        y_train_pred += train_Y_old
    denormalized_train_y, denormalized_train_y_pred = denormalize_xgboost_predictions(
        train_Y, y_train_pred, data_module.factors, y_key=y_key
    )
    train_mse = np.mean((denormalized_train_y - denormalized_train_y_pred) ** 2)
    train_rmse = np.sqrt(train_mse)
    train_mae = np.mean(np.abs(denormalized_train_y - denormalized_train_y_pred))
    print(f"Training RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}")

    # Evaluate on validation set
    print("Evaluating on validation set...")
    y_pred = model.predict(val_X)
    if xgboost_config.get("mode", "default") == "diff":
        y_pred += val_Y_old
    denormalized_y, denormalized_y_pred = denormalize_xgboost_predictions(
        val_Y, y_pred, data_module.factors, y_key=y_key
    )
    val_mse = np.mean((denormalized_y - denormalized_y_pred) ** 2)
    val_rmse = np.sqrt(val_mse)
    val_mae = np.mean(np.abs(denormalized_y - denormalized_y_pred))
    print(f"Validation RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}")

    # Evaluate on test set
    print("Evaluating on test set...")
    y_test_pred = model.predict(test_X)
    if xgboost_config.get("mode", "default") == "diff":
        y_test_pred += test_Y_old
    denormalized_test_y, denormalized_test_y_pred = denormalize_xgboost_predictions(
        test_Y, y_test_pred, data_module.factors, y_key=y_key
    )
    test_mse = np.mean((denormalized_test_y - denormalized_test_y_pred) ** 2)
    test_rmse = np.sqrt(test_mse)
    test_mae = np.mean(np.abs(denormalized_test_y - denormalized_test_y_pred))
    print(f"Test RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")

    # 베이스라인
    print("Calculating baseline performance...")
    y_test_pred = test_Y_old
    denormalized_test_y, denormalized_test_y_pred = denormalize_xgboost_predictions(
        test_Y, y_test_pred, data_module.factors, y_key=y_key
    )
    test_mse = np.mean((denormalized_test_y - denormalized_test_y_pred) ** 2)
    test_rmse = np.sqrt(test_mse)
    test_mae = np.mean(np.abs(denormalized_test_y - denormalized_test_y_pred))
    print(f"Baseline-Test RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")

    # 모델 세이브
    if args.save_checkpoints:
        if not os.path.exists(args.expname):
            os.makedirs(args.expname)
        print("Saving model...")
        model_save_path = os.path.join(
            args.expname, f"xgboost_model-val-rmse{test_rmse:.4f}.json"
        )
        model.save_model(model_save_path)
        print(f"Model saved to {model_save_path}")


###########################
if __name__ == "__main__":

    args = get_args()
    if args.use_xgboost:
        train_xgboost(args)
    else:
        train_lightning(args)
