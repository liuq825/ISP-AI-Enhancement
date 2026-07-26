from __future__ import annotations

import json
import random
from pathlib import Path
from time import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from isp_ai_enhancement.config import load_yaml
from isp_ai_enhancement.data.context import ContextBuilder, ContextConfig
from isp_ai_enhancement.data.dataset import RawPairDataset
from isp_ai_enhancement.distillation import FeatureDistiller
from isp_ai_enhancement.export import load_checkpoint_state
from isp_ai_enhancement.losses import LossWeights, RawRestorationLoss
from isp_ai_enhancement.metrics import psnr
from isp_ai_enhancement.models.factory import build_model_from_file


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict[str, Any],
    distiller: nn.Module | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config,
        "created_unix": int(time()),
    }
    if distiller is not None:
        payload["distiller_state"] = distiller.state_dict()
    torch.save(payload, path)


@torch.inference_mode()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    values: list[float] = []
    for batch in loader:
        inputs = batch["input"].to(device)
        target = batch["target"].to(device)
        enhanced = torch.clamp(inputs[:, :4] + model(inputs), 0.0, 1.0)
        values.append(float(psnr(enhanced, target).item()))
    return sum(values) / max(1, len(values))


def train_from_config(path: str | Path) -> Path:
    config_path = Path(path)
    config = load_yaml(config_path)
    seed = int(config.get("seed", 20260726))
    _seed_everything(seed)
    device_name = str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    model_config_value = config.get("model_config", config.get("student_config"))
    if model_config_value is None:
        raise ValueError("training config must define model_config or student_config")
    model_config = Path(str(model_config_value))
    if not model_config.is_absolute():
        model_config = config_path.parent.parent / model_config
    manifest = Path(str(config["manifest"]))
    if not manifest.is_absolute():
        manifest = config_path.parent.parent / manifest
    output_dir = Path(str(config.get("output_dir", "runs/train")))
    if not output_dir.is_absolute():
        output_dir = config_path.parent.parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    builder = ContextBuilder(
        ContextConfig(camera_embeddings={"smoke_sensor": (0.0, 0.0, 0.0, 0.0)})
    )
    crop_size = int(config["crop_size"]) if config.get("crop_size") else None
    train_data = RawPairDataset(
        manifest,
        split="train",
        context_builder=builder,
        crop_size=crop_size,
        augment=True,
    )
    val_data = RawPairDataset(
        manifest,
        split="val",
        context_builder=builder,
        crop_size=crop_size,
        augment=False,
    )
    loader_options = {
        "batch_size": int(config.get("batch_size", 1)),
        "num_workers": int(config.get("num_workers", 0)),
    }
    train_loader = DataLoader(train_data, shuffle=True, **loader_options)
    val_loader = DataLoader(val_data, shuffle=False, **loader_options)
    model = build_model_from_file(model_config).to(device)
    teacher: nn.Module | None = None
    distiller: FeatureDistiller | None = None
    teacher_feature_weight = 0.0
    if config.get("teacher_config") or config.get("teacher_checkpoint"):
        if not config.get("teacher_config") or not config.get("teacher_checkpoint"):
            raise ValueError("teacher_config and teacher_checkpoint must be provided together")
        teacher_config = Path(str(config["teacher_config"]))
        teacher_checkpoint = Path(str(config["teacher_checkpoint"]))
        if not teacher_config.is_absolute():
            teacher_config = config_path.parent.parent / teacher_config
        if not teacher_checkpoint.is_absolute():
            teacher_checkpoint = config_path.parent.parent / teacher_checkpoint
        teacher = build_model_from_file(teacher_config).to(device)
        teacher.load_state_dict(load_checkpoint_state(teacher_checkpoint), strict=True)
        teacher.eval()
        teacher.requires_grad_(False)
        loss_settings = dict(config.get("loss", {}))
        feature_keys = tuple(loss_settings.get("feature_keys", ("enc2", "enc4", "middle", "dec2")))
        distiller = FeatureDistiller(
            student_width=model.width,
            teacher_width=teacher.width,
            keys=feature_keys,
        ).to(device)
        teacher_feature_weight = float(loss_settings.get("teacher_feature", 0.10))
    optimized_parameters = list(model.parameters())
    if distiller is not None:
        optimized_parameters.extend(distiller.parameters())
    optimizer = torch.optim.AdamW(
        optimized_parameters,
        lr=float(config.get("learning_rate", 2e-4)),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    loss_values = dict(config.get("loss", {}))
    criterion = RawRestorationLoss(
        LossWeights(
            charbonnier=float(loss_values.get("charbonnier", 1.0)),
            gradient=float(loss_values.get("gradient", 0.1)),
            color=float(loss_values.get("color", 0.05)),
            teacher_output=float(loss_values.get("teacher_output", 0.0)),
        )
    )
    epochs = int(config.get("epochs", 1))
    log_every = int(config.get("log_every", 20))
    history_path = output_dir / "history.jsonl"
    global_step = 0
    with history_path.open("w", encoding="utf-8", newline="\n") as history:
        for epoch in range(1, epochs + 1):
            model.train()
            for batch in train_loader:
                inputs = batch["input"].to(device)
                target = batch["target"].to(device)
                valid_mask = inputs[:, 15:16]
                optimizer.zero_grad(set_to_none=True)
                residual, student_features = model.forward_features(inputs)
                enhanced = torch.clamp(inputs[:, :4] + residual, 0.0, 1.0)
                teacher_enhanced = None
                teacher_features = None
                if teacher is not None:
                    with torch.no_grad():
                        teacher_residual, teacher_features = teacher.forward_features(inputs)
                        teacher_enhanced = torch.clamp(inputs[:, :4] + teacher_residual, 0.0, 1.0)
                loss, terms = criterion(
                    enhanced,
                    target,
                    mask=valid_mask,
                    teacher_enhanced=teacher_enhanced,
                )
                if distiller is not None and teacher_features is not None:
                    feature_loss = distiller(student_features, teacher_features)
                    terms["teacher_feature"] = feature_loss
                    loss = loss + teacher_feature_weight * feature_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                global_step += 1
                if global_step % log_every == 0:
                    record = {
                        "epoch": epoch,
                        "step": global_step,
                        "loss": float(loss.detach().item()),
                        **{
                            f"loss_{name}": float(value.detach().item())
                            for name, value in terms.items()
                        },
                    }
                    history.write(json.dumps(record, sort_keys=True) + "\n")
                    history.flush()
            validation_psnr = _evaluate(model, val_loader, device)
            history.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "step": global_step,
                        "validation_psnr": validation_psnr,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            _save_checkpoint(
                output_dir / f"epoch_{epoch:04d}.pt",
                model,
                optimizer,
                epoch,
                config,
                distiller,
            )
    return output_dir / f"epoch_{epochs:04d}.pt"
