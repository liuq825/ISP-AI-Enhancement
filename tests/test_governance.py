"""验证研发与生产两级数据许可治理门禁。"""

from pathlib import Path

import yaml

from isp_ai_enhancement.data.context import ContextConfig
from isp_ai_enhancement.data.governance import (
    enforce_data_policy,
    validate_data_policy,
)
from isp_ai_enhancement.data.manifest import ManifestRecord


def _record(dataset_id: str, sensor_id: str = "sensor") -> ManifestRecord:
    """构造只包含治理测试所需字段的最小清单记录。"""

    return ManifestRecord(
        sample_id=f"{dataset_id}_sample",
        dataset_id=dataset_id,
        input_path="input.npz",
        target_path="target.npz",
        split="train",
        sensor_id=sensor_id,
        mode="single",
        session_id="session",
        scene_id="scene",
        iso_bucket="low",
        metadata={},
    )


def _write_yaml(path: Path, value: dict) -> Path:
    """写出临时 YAML 并返回路径，减少测试样板代码。"""

    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def _catalog(tmp_path: Path) -> Path:
    """创建覆盖冒烟、目标传感器和非商用基准的测试目录。"""

    return _write_yaml(
        tmp_path / "datasets.yaml",
        {
            "catalog_version": 2,
            "datasets": [
                {
                    "id": "synthetic_smoke",
                    "role": "synthetic_smoke",
                    "allowed_uses": ["smoke"],
                    "production_default": "prohibited",
                },
                {
                    "id": "target",
                    "role": "target_sensor",
                    "allowed_uses": ["research", "production"],
                    "production_default": "review_required",
                },
                {
                    "id": "dnd",
                    "role": "public_benchmark",
                    "allowed_uses": ["research"],
                    "production_default": "prohibited",
                },
            ],
        },
    )


def _approval(tmp_path: Path) -> Path:
    """创建字段完整的目标传感器生产审批记录。"""

    return _write_yaml(
        tmp_path / "approval.yaml",
        {
            "approval_version": 1,
            "approved_datasets": {
                "target": {
                    "approved_uses": ["production"],
                    "reviewer": "legal@example.test",
                    "approved_on": "2026-07-26",
                    "license_snapshot_sha256": "a" * 64,
                    "provenance_snapshot_sha256": "b" * 64,
                }
            },
        },
    )


def test_smoke_policy_allows_only_catalogued_smoke_data(tmp_path: Path) -> None:
    """已登记的合成数据应仅能通过冒烟用途检查。"""

    assert (
        validate_data_policy(
            [_record("synthetic_smoke")],
            catalog_path=_catalog(tmp_path),
            purpose="smoke",
        )
        == []
    )


def test_production_requires_approval_and_target_sensor(tmp_path: Path) -> None:
    """生产用途缺少审批文件时必须被阻止。"""

    errors = validate_data_policy(
        [_record("target")],
        catalog_path=_catalog(tmp_path),
        purpose="production",
    )
    assert "production training requires a data approval file" in errors


def test_commercial_grade_research_does_not_require_production_approval(
    tmp_path: Path,
) -> None:
    """商用质量研发阶段不应被未来生产审批流程错误阻塞。"""

    catalog = _catalog(tmp_path)
    value = yaml.safe_load(catalog.read_text(encoding="utf-8"))
    value["datasets"][1]["allowed_uses"].append("commercial_grade")
    _write_yaml(catalog, value)
    assert (
        validate_data_policy(
            [_record("target")],
            catalog_path=catalog,
            purpose="commercial_grade",
        )
        == []
    )


def test_production_accepts_complete_target_sensor_approval(tmp_path: Path) -> None:
    """目标传感器数据与完整审批应通过生产门禁。"""

    context = ContextConfig(camera_embeddings={"sensor": (0.0, 0.0, 0.0, 0.0)})
    enforce_data_policy(
        [_record("target")],
        catalog_path=_catalog(tmp_path),
        purpose="production",
        approval_path=_approval(tmp_path),
        context_config=context,
    )


def test_production_rejects_noncommercial_dataset(tmp_path: Path) -> None:
    """目录明确禁止生产的数据不得被审批文件绕过。"""

    errors = validate_data_policy(
        [_record("dnd")],
        catalog_path=_catalog(tmp_path),
        purpose="production",
        approval_path=_approval(tmp_path),
    )
    assert any("not allowed for production" in error for error in errors)
    assert any("prohibited for production" in error for error in errors)
