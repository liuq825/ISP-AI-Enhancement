"""训练、验证、蒸馏、检查点保存和数据门禁的一体化执行引擎。

训练入口先验证清单、许可用途和传感器上下文，再创建 DataLoader。模型预测
四通道残差并与原始 packed RAW 相加；可选教师模型提供输出和中间特征蒸馏。
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from time import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from isp_ai_enhancement.config import load_yaml
from isp_ai_enhancement.data.context import ContextBuilder, load_context_config
from isp_ai_enhancement.data.dataset import RawPairDataset
from isp_ai_enhancement.data.governance import enforce_data_policy
from isp_ai_enhancement.data.manifest import read_manifest, validate_manifest
from isp_ai_enhancement.distillation import FeatureDistiller
from isp_ai_enhancement.export import load_checkpoint_state
from isp_ai_enhancement.losses import LossWeights, RawRestorationLoss
from isp_ai_enhancement.metrics import psnr_per_sample
from isp_ai_enhancement.models.factory import build_model_from_file
from isp_ai_enhancement.quantization.fake_quant import (
    QATReport,
    prepare_qat,
    set_observer_enabled,
)


def _seed_everything(seed: int) -> None:
    """固定 Python、NumPy、CPU 与 CUDA 随机源，提高实验可复现性。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_worker(_worker_id: int) -> None:
    """用 PyTorch 分配的 worker 种子同步初始化 NumPy 与 Python 随机源。"""

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LRScheduler | None,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    best_validation_psnr: float,
    config: dict[str, Any],
    loader_generator: torch.Generator,
    distiller: nn.Module | None = None,
) -> None:
    """原子保存可精确恢复的版本化检查点与全部随机状态。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 2,
        "epoch": epoch,
        "global_step": global_step,
        "best_validation_psnr": best_validation_psnr,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "loader_generator_state": loader_generator.get_state(),
        "config": config,
        "created_unix": int(time()),
    }
    if distiller is not None:
        payload["distiller_state"] = distiller.state_dict()
    # 先写同目录临时文件再替换，进程中断不会留下一个貌似有效的半截 checkpoint。
    temporary = path.with_name(f"{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _restore_training_state(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LRScheduler | None,
    scaler: torch.amp.GradScaler,
    loader_generator: torch.Generator,
    distiller: nn.Module | None,
) -> tuple[int, int, float]:
    """恢复模型、优化器、调度器、AMP 与随机状态，返回下一轮训练位置。"""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or int(payload.get("format_version", 0)) < 2:
        raise ValueError("resume checkpoint must use training format_version >= 2")
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    saved_scheduler = payload.get("scheduler_state")
    if (scheduler is None) != (saved_scheduler is None):
        raise ValueError("resume checkpoint scheduler configuration does not match")
    if scheduler is not None:
        scheduler.load_state_dict(saved_scheduler)
    scaler.load_state_dict(payload.get("scaler_state", {}))
    saved_distiller = payload.get("distiller_state")
    if (distiller is None) != (saved_distiller is None):
        raise ValueError("resume checkpoint distillation configuration does not match")
    if distiller is not None:
        distiller.load_state_dict(saved_distiller, strict=True)

    random.setstate(payload["python_rng_state"])
    np.random.set_state(payload["numpy_rng_state"])
    torch.set_rng_state(payload["torch_rng_state"])
    if torch.cuda.is_available() and payload.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
    loader_generator.set_state(payload["loader_generator_state"])
    return (
        int(payload["epoch"]) + 1,
        int(payload["global_step"]),
        float(payload["best_validation_psnr"]),
    )


@torch.inference_mode()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
) -> float:
    """在验证集上计算逐样本平均 packed RAW PSNR，不创建梯度图。"""

    model.eval()
    values: list[float] = []
    for batch in loader:
        inputs = batch["input"].to(device)
        target = batch["target"].to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            enhanced = torch.clamp(inputs[:, :4] + model(inputs), 0.0, 1.0)
        values.extend(
            float(value)
            for value in psnr_per_sample(
                enhanced, target, inputs[:, 15:16]
            ).cpu().tolist()
        )
    return sum(values) / max(1, len(values))


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    epochs: int,
) -> LRScheduler | None:
    """按配置创建 epoch 级学习率调度器；省略配置时保持常数学习率。"""

    settings = config.get("scheduler")
    if settings is None:
        return None
    if not isinstance(settings, dict):
        raise ValueError("scheduler must be a mapping")
    scheduler_type = str(settings.get("type", "cosine")).lower()
    if scheduler_type != "cosine":
        raise ValueError("only cosine scheduler is currently supported")
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(settings.get("t_max", epochs)),
        eta_min=float(settings.get("eta_min", 1e-6)),
    )


def train_from_config(path: str | Path) -> Path:
    """按 YAML 配置完成训练并返回最后一个检查点路径。

    该入口执行“先治理、后训练”：只有清单文件、数据用途和相机嵌入全部
    通过预检才会创建输出目录并开始优化，避免不合规数据产生不可追溯权重。
    """

    config_path = Path(path)
    config = load_yaml(config_path)
    seed = int(config.get("seed", 20260726))
    _seed_everything(seed)
    device_name = str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    amp_enabled = bool(config.get("amp", False))
    if amp_enabled and device.type != "cuda":
        raise ValueError("float16 AMP is supported only when training on CUDA")
    initial_value = config.get("initial_checkpoint")
    resume_value = config.get("resume_checkpoint")
    if initial_value is not None and resume_value is not None:
        raise ValueError("initial_checkpoint and resume_checkpoint are mutually exclusive")
    model_config_value = config.get("model_config", config.get("student_config"))
    if model_config_value is None:
        raise ValueError("training config must define model_config or student_config")
    model_config = Path(str(model_config_value))
    if not model_config.is_absolute():
        model_config = config_path.parent.parent / model_config
    manifest = Path(str(config["manifest"]))
    if not manifest.is_absolute():
        manifest = config_path.parent.parent / manifest
    context_config_value = config.get("context_config")
    if context_config_value is None:
        raise ValueError("training config must define context_config")
    context_config_path = Path(str(context_config_value))
    if not context_config_path.is_absolute():
        context_config_path = config_path.parent.parent / context_config_path
    context_config = load_context_config(context_config_path)
    records = read_manifest(manifest)
    manifest_errors = validate_manifest(records, root=manifest.parent)
    if manifest_errors:
        formatted = "\n".join(f"- {error}" for error in manifest_errors)
        raise ValueError(f"manifest validation failed:\n{formatted}")

    data_policy = config.get("data_policy")
    if not isinstance(data_policy, dict):
        raise ValueError("training config must define a data_policy mapping")
    purpose = str(data_policy.get("purpose", ""))
    catalog_value = data_policy.get("catalog")
    if not purpose or catalog_value is None:
        raise ValueError("data_policy must define purpose and catalog")
    catalog_path = Path(str(catalog_value))
    if not catalog_path.is_absolute():
        catalog_path = config_path.parent.parent / catalog_path
    approval_value = data_policy.get("approval")
    approval_path = Path(str(approval_value)) if approval_value is not None else None
    if approval_path is not None and not approval_path.is_absolute():
        approval_path = config_path.parent.parent / approval_path
    # 数据许可与相机上下文属于训练硬门禁，不能仅依赖人工检查文档。
    enforce_data_policy(
        records,
        catalog_path=catalog_path,
        purpose=purpose,
        approval_path=approval_path,
        context_config=context_config,
    )
    output_dir = Path(str(config.get("output_dir", "runs/train")))
    if not output_dir.is_absolute():
        output_dir = config_path.parent.parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    builder = ContextBuilder(context_config)
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
    num_workers = int(config.get("num_workers", 0))
    train_generator = torch.Generator().manual_seed(seed)
    validation_generator = torch.Generator().manual_seed(seed + 1)
    loader_options = {
        "batch_size": int(config.get("batch_size", 1)),
        "num_workers": num_workers,
        "worker_init_fn": _seed_worker,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        train_data,
        shuffle=True,
        generator=train_generator,
        **loader_options,
    )
    val_loader = DataLoader(
        val_data,
        shuffle=False,
        generator=validation_generator,
        **loader_options,
    )
    model = build_model_from_file(model_config).to(device)
    if initial_value is not None:
        initial_path = Path(str(initial_value))
        if not initial_path.is_absolute():
            initial_path = config_path.parent.parent / initial_path
        model.load_state_dict(load_checkpoint_state(initial_path), strict=True)

    qat_report: QATReport | None = None
    qat_observer_warmup_steps: int | None = None
    qat_config_value = config.get("qat_config")
    if qat_config_value is not None:
        qat_config_path = Path(str(qat_config_value))
        if not qat_config_path.is_absolute():
            qat_config_path = config_path.parent.parent / qat_config_path
        qat_settings = load_yaml(qat_config_path).get("qat")
        if not isinstance(qat_settings, dict):
            raise ValueError("qat_config must contain a 'qat' mapping")
        qat_observer_warmup_steps = int(
            qat_settings.get("observer_warmup_steps", 10_000)
        )
        if qat_observer_warmup_steps <= 0:
            raise ValueError("QAT observer_warmup_steps must be positive")
        exclude_modules = qat_settings.get("exclude_modules", ("intro", "ending"))
        if not isinstance(exclude_modules, (list, tuple)) or not all(
            isinstance(value, str) for value in exclude_modules
        ):
            raise ValueError("QAT exclude_modules must be a list of module names")
        qat_report = prepare_qat(
            model,
            activation_bits=int(qat_settings.get("activation_bits", 8)),
            weight_bits=int(qat_settings.get("weight_bits", 8)),
            observer_momentum=float(qat_settings.get("observer_momentum", 0.95)),
            exclude_modules=tuple(exclude_modules),
        )
    teacher: nn.Module | None = None
    distiller: FeatureDistiller | None = None
    teacher_feature_weight = 0.0
    # 教师配置和权重必须成对出现，避免误以为已经启用知识蒸馏。
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
    save_every_epochs = int(config.get("save_every_epochs", 1))
    if epochs <= 0 or log_every <= 0 or save_every_epochs <= 0:
        raise ValueError("epochs, log_every, and save_every_epochs must be positive")
    scheduler = _build_scheduler(optimizer, config, epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    history_path = output_dir / "history.jsonl"
    global_step = 0
    start_epoch = 1
    best_validation_psnr = float("-inf")
    if resume_value is not None:
        resume_path = Path(str(resume_value))
        if not resume_path.is_absolute():
            resume_path = config_path.parent.parent / resume_path
        start_epoch, global_step, best_validation_psnr = _restore_training_state(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            loader_generator=train_generator,
            distiller=distiller,
        )
    if start_epoch > epochs:
        raise ValueError(
            f"resume checkpoint already completed epoch {start_epoch - 1}, "
            f"but configured epochs is {epochs}"
        )

    history_mode = "a" if resume_value is not None else "w"
    with history_path.open(history_mode, encoding="utf-8", newline="\n") as history:
        if qat_report is not None and history_mode == "w":
            history.write(
                json.dumps(
                    {
                        "event": "qat_prepared",
                        "converted_convolutions": qat_report.converted_convolutions,
                        "excluded_convolutions": qat_report.excluded_convolutions,
                        "simulated_int8_layer_ratio": qat_report.simulated_int8_ratio,
                        "simulated_int8_weight_ratio": (
                            qat_report.simulated_int8_weight_ratio
                        ),
                        "observer_warmup_steps": qat_observer_warmup_steps,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        for epoch in range(start_epoch, epochs + 1):
            model.train()
            for batch in train_loader:
                if (
                    qat_observer_warmup_steps is not None
                    and global_step >= qat_observer_warmup_steps
                ):
                    # 热身结束后冻结统计尺度，后续只优化权重并注入固定量化误差。
                    set_observer_enabled(model, False)
                inputs = batch["input"].to(device)
                target = batch["target"].to(device)
                # 第 16 通道是输入契约定义的有效像素掩码，只参与损失加权。
                valid_mask = inputs[:, 15:16]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    residual, student_features = model.forward_features(inputs)
                    enhanced = torch.clamp(inputs[:, :4] + residual, 0.0, 1.0)
                    teacher_enhanced = None
                    teacher_features = None
                    if teacher is not None:
                        # 教师固定为推理态；只让学生和特征适配器接收梯度。
                        with torch.no_grad():
                            teacher_residual, teacher_features = teacher.forward_features(inputs)
                            teacher_enhanced = torch.clamp(
                                inputs[:, :4] + teacher_residual, 0.0, 1.0
                            )
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
                scaler.scale(loss).backward()
                # 裁剪前必须把 AMP 梯度还原到真实尺度，否则阈值 1.0 没有物理意义。
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(optimized_parameters, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                global_step += 1
                if global_step % log_every == 0:
                    record = {
                        "epoch": epoch,
                        "step": global_step,
                        "loss": float(loss.detach().item()),
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                        "grad_scale": float(scaler.get_scale()),
                        **{
                            f"loss_{name}": float(value.detach().item())
                            for name, value in terms.items()
                        },
                    }
                    history.write(json.dumps(record, sort_keys=True) + "\n")
                    history.flush()
            validation_psnr = _evaluate(model, val_loader, device, amp_enabled)
            if scheduler is not None:
                # 按官方建议在本轮 optimizer.step() 全部完成后推进 epoch 级调度器。
                scheduler.step()
            is_best = validation_psnr > best_validation_psnr
            best_validation_psnr = max(best_validation_psnr, validation_psnr)
            history.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "step": global_step,
                        "validation_psnr": validation_psnr,
                        "best_validation_psnr": best_validation_psnr,
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                        "is_best": is_best,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            history.flush()
            if epoch % save_every_epochs == 0 or epoch == epochs:
                _save_checkpoint(
                    output_dir / f"epoch_{epoch:04d}.pt",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    global_step,
                    best_validation_psnr,
                    config,
                    train_generator,
                    distiller,
                )
            if is_best:
                _save_checkpoint(
                    output_dir / "best.pt",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    global_step,
                    best_validation_psnr,
                    config,
                    train_generator,
                    distiller,
                )
    return output_dir / f"epoch_{epochs:04d}.pt"
