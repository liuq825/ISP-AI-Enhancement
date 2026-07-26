from pathlib import Path

from isp_ai_enhancement.data.manifest import ManifestRecord, read_manifest, validate_manifest
from isp_ai_enhancement.data.synthetic import generate_smoke_dataset


def test_smoke_dataset_and_manifest(tmp_path: Path) -> None:
    manifest = generate_smoke_dataset(tmp_path / "smoke", samples=8, height=32, width=32)
    records = read_manifest(manifest)
    assert len(records) == 8
    assert validate_manifest(records, root=manifest.parent) == []


def test_manifest_detects_split_leakage() -> None:
    base = dict(
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
