"""验证已训练 checkpoint 的物理剪枝产物与双后端一致性。"""

from pathlib import Path

import torch

from isp_ai_enhancement.export import load_checkpoint_state
from isp_ai_enhancement.models.nafnet import ExpansionSpec, NAFNetRaw
from isp_ai_enhancement.pruning.physical import physical_prune
from isp_ai_enhancement.pruning.torch_pruning_adapter import (
    torch_pruning_physical_prune,
)
from isp_ai_enhancement.pruning.workflow import prune_checkpoint


def _write_model_config(path: Path, enc1_hidden: int) -> Path:
    """写出仅第一编码块扩展宽度可变的极小 NAFNet 配置。"""

    path.write_text(
        "model:\n"
        "  name: nafnet_raw\n"
        "  input_channels: 16\n"
        "  output_channels: 4\n"
        "  width: 4\n"
        "  encoder_blocks: [1, 1, 1, 1]\n"
        "  middle_blocks: 1\n"
        "  decoder_blocks: [1, 1, 1, 1]\n"
        "  expansion_spec:\n"
        f"    enc1: [{enc1_hidden}]\n"
        "    enc2: [8]\n"
        "    enc3: [16]\n"
        "    enc4: [32]\n"
        "    middle: [64]\n"
        "    dec1: [32]\n"
        "    dec2: [16]\n"
        "    dec3: [8]\n"
        "    dec4: [4]\n",
        encoding="utf-8",
    )
    return path


def test_manual_and_torch_pruning_produce_identical_weights() -> None:
    """相同重要性排序下，手工重建与 DepGraph 输出必须逐元素一致。"""

    torch.manual_seed(31)
    source = NAFNetRaw(
        width=4,
        encoder_blocks=(1, 1, 1, 1),
        middle_blocks=1,
        decoder_blocks=(1, 1, 1, 1),
    ).eval()
    values = source.expansion_spec.as_dict()
    values["enc1"] = [3]
    target_spec = ExpansionSpec.from_mapping(values)
    manual, _manual_report = physical_prune(source, target_spec)
    dependency, _dependency_report, _backend = torch_pruning_physical_prune(
        source,
        target_spec,
    )
    for name, value in manual.state_dict().items():
        torch.testing.assert_close(
            value,
            dependency.state_dict()[name],
            rtol=0,
            atol=0,
        )


def test_prune_checkpoint_is_rebuildable_from_target_config(tmp_path: Path) -> None:
    """剪枝产物应能被目标 YAML 严格加载，并包含完整来源哈希。"""

    torch.manual_seed(37)
    source = NAFNetRaw(
        width=4,
        encoder_blocks=(1, 1, 1, 1),
        middle_blocks=1,
        decoder_blocks=(1, 1, 1, 1),
    )
    source_config = _write_model_config(tmp_path / "source.yaml", 4)
    target_config = _write_model_config(tmp_path / "target.yaml", 3)
    source_checkpoint = tmp_path / "source.pt"
    torch.save({"model_state": source.state_dict()}, source_checkpoint)
    output, metadata = prune_checkpoint(
        source_config=source_config,
        source_checkpoint=source_checkpoint,
        target_config=target_config,
        output=tmp_path / "pruned.pt",
    )
    state = load_checkpoint_state(output)
    assert any(value.numel() for value in state.values())
    assert metadata["target_parameters"] < metadata["source_parameters"]
    assert metadata["backend"]["version"] == "1.6.1"
    assert output.with_suffix(".pt.manifest.json").is_file()
