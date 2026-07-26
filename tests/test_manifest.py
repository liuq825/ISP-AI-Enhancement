"""验证数据清单生成、文件检查与场景泄漏检测。"""

from pathlib import Path

from isp_ai_enhancement.data.manifest import ManifestRecord, read_manifest, validate_manifest
from isp_ai_enhancement.data.synthetic import generate_smoke_dataset


def test_smoke_dataset_and_manifest(tmp_path: Path) -> None:
    """合成冒烟集应产生完整且文件可解析的清单。"""

    manifest = generate_smoke_dataset(tmp_path / "smoke", samples=8, height=32, width=32)
    records = read_manifest(manifest)
    assert len(records) == 8
    assert validate_manifest(records, root=manifest.parent) == []


def test_manifest_detects_split_leakage() -> None:
    """同一会话和场景出现在训练与测试集合时必须报错。"""

    base = dict(
        dataset_id="dataset",
        input_path="input.npz",
        target_path="target.npz",
        sensor_id="sensor",
        mode="single",
        session_id="same",
        scene_id="same",
        iso_bucket="low",
        metadata={},
    )
    records = [
        ManifestRecord(sample_id="a", split="train", **base),
        ManifestRecord(sample_id="b", split="test", **base),
    ]
    errors = validate_manifest(records, require_files=False)
    assert any("leakage" in error for error in errors)
