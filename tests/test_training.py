"""验证训练检查点、余弦调度与随机状态恢复的确定性。"""

from pathlib import Path

import torch
import yaml

from isp_ai_enhancement.data.synthetic import generate_smoke_dataset
from isp_ai_enhancement.training.engine import train_from_config


def _write_training_config(
    path: Path,
    *,
    manifest: Path,
    model_config: Path,
    output_dir: Path,
    resume_checkpoint: Path | None = None,
) -> Path:
    """写出可在 CPU 快速运行的两轮训练配置。"""

    value = {
        "seed": 20260726,
        "device": "cpu",
        "model_config": str(model_config),
        "context_config": str(Path("configs/context.yaml").resolve()),
        "manifest": str(manifest),
        "output_dir": str(output_dir),
        "data_policy": {
            "purpose": "smoke",
            "catalog": str(Path("resources/datasets.yaml").resolve()),
        },
        "epochs": 2,
        "batch_size": 2,
        "num_workers": 0,
        "crop_size": 32,
        "learning_rate": 0.0002,
        "amp": False,
        "log_every": 1,
        "save_every_epochs": 1,
        "scheduler": {"type": "cosine", "t_max": 2, "eta_min": 0.000001},
        "loss": {"charbonnier": 1.0, "gradient": 0.1, "color": 0.05},
    }
    if resume_checkpoint is not None:
        value["resume_checkpoint"] = str(resume_checkpoint)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def test_resume_reproduces_uninterrupted_second_epoch(tmp_path: Path) -> None:
    """从第一轮 checkpoint 恢复后的第二轮权重应与连续训练逐元素一致。"""

    manifest = generate_smoke_dataset(
        tmp_path / "data",
        samples=8,
        height=32,
        width=32,
    )
    model_config = tmp_path / "model.yaml"
    model_config.write_text(
        "model:\n"
        "  name: nafnet_raw\n"
        "  input_channels: 16\n"
        "  output_channels: 4\n"
        "  width: 4\n"
        "  encoder_blocks: [1, 1, 1, 1]\n"
        "  middle_blocks: 1\n"
        "  decoder_blocks: [1, 1, 1, 1]\n"
        "  expansion_spec: baseline\n",
        encoding="utf-8",
    )
    continuous_dir = tmp_path / "continuous"
    continuous_config = _write_training_config(
        tmp_path / "continuous.yaml",
        manifest=manifest,
        model_config=model_config,
        output_dir=continuous_dir,
    )
    continuous_final = train_from_config(continuous_config)
    first_epoch = continuous_dir / "epoch_0001.pt"

    resumed_dir = tmp_path / "resumed"
    resumed_config = _write_training_config(
        tmp_path / "resumed.yaml",
        manifest=manifest,
        model_config=model_config,
        output_dir=resumed_dir,
        resume_checkpoint=first_epoch,
    )
    resumed_final = train_from_config(resumed_config)
    continuous = torch.load(continuous_final, map_location="cpu", weights_only=False)
    resumed = torch.load(resumed_final, map_location="cpu", weights_only=False)

    assert resumed["format_version"] == 2
    assert resumed["epoch"] == continuous["epoch"] == 2
    assert resumed["global_step"] == continuous["global_step"]
    assert resumed["scheduler_state"] == continuous["scheduler_state"]
    for name, value in continuous["model_state"].items():
        torch.testing.assert_close(value, resumed["model_state"][name], rtol=0, atol=0)
