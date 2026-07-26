"""数据许可、来源与相机上下文覆盖的训练前门禁。

本项目当前目标是达到商用模型技术要求，而非直接将数据或模型投入生产，
因此常规研发使用 ``commercial_grade``。保留更严格的 ``production`` 模式，
用于未来正式量产时校验审批人、许可快照和目标传感器数据。
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from isp_ai_enhancement.config import load_yaml

from .context import ContextConfig
from .manifest import ManifestRecord

_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_PURPOSES = {"smoke", "research", "commercial_grade", "production"}


def _dataset_index(catalog: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """把数据集目录转换为按 ID 索引的映射，并收集结构错误。"""

    errors: list[str] = []
    raw_entries = catalog.get("datasets")
    if not isinstance(raw_entries, list):
        return {}, ["dataset catalog must contain a 'datasets' list"]
    index: dict[str, dict[str, Any]] = {}
    for position, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, dict):
            errors.append(f"dataset catalog entry {position} is not a mapping")
            continue
        dataset_id = str(raw_entry.get("id", "")).strip()
        if not dataset_id:
            errors.append(f"dataset catalog entry {position} has no id")
            continue
        if dataset_id in index:
            errors.append(f"duplicate dataset catalog id: {dataset_id}")
            continue
        index[dataset_id] = raw_entry
    return index, errors


def _approval_index(approval: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """解析正式生产审批文件，确保每个条目都是结构化映射。"""

    raw_entries = approval.get("approved_datasets")
    if not isinstance(raw_entries, Mapping):
        return {}, ["approval file must contain an 'approved_datasets' mapping"]
    errors: list[str] = []
    index: dict[str, dict[str, Any]] = {}
    for dataset_id, raw_entry in raw_entries.items():
        if not isinstance(raw_entry, dict):
            errors.append(f"approval for {dataset_id!r} is not a mapping")
            continue
        index[str(dataset_id)] = raw_entry
    return index, errors


def _validate_production_approval(dataset_id: str, value: Mapping[str, Any]) -> list[str]:
    """验证一个数据集的生产审批字段及两份证据快照哈希。"""

    errors: list[str] = []
    approved_uses = value.get("approved_uses", [])
    if not isinstance(approved_uses, list) or "production" not in approved_uses:
        errors.append(f"{dataset_id}: approval does not include production use")
    for field_name in ("reviewer", "approved_on"):
        if not str(value.get(field_name, "")).strip():
            errors.append(f"{dataset_id}: approval is missing {field_name}")
    for field_name in ("license_snapshot_sha256", "provenance_snapshot_sha256"):
        digest = str(value.get(field_name, ""))
        if not _SHA256.fullmatch(digest):
            errors.append(f"{dataset_id}: {field_name} must be a 64-digit SHA256")
    return errors


def validate_context_coverage(
    records: Iterable[ManifestRecord],
    context_config: ContextConfig,
) -> list[str]:
    """要求训练涉及的每个传感器都存在版本化相机嵌入。

    若清单内同时写入了嵌入值，则必须与注册表完全相同，防止相同
    ``sensor_id`` 在不同样本中被赋予不同语义。
    """

    errors: list[str] = []
    for record in records:
        registered = context_config.camera_embeddings.get(record.sensor_id)
        if registered is None:
            errors.append(
                f"{record.sample_id}: sensor {record.sensor_id!r} is absent from context config"
            )
            continue
        inline = record.metadata.get("camera_embedding")
        if inline is not None:
            try:
                inline_values = tuple(float(item) for item in inline)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                errors.append(f"{record.sample_id}: invalid inline camera_embedding")
                continue
            if inline_values != tuple(float(item) for item in registered):
                errors.append(
                    f"{record.sample_id}: inline camera_embedding differs from context registry"
                )
    return errors


def _required_strings(
    requirements: Mapping[str, Any],
    field_name: str,
    errors: list[str],
) -> set[str]:
    """读取一个必需字符串列表；格式错误时记录门禁错误而不是猜测。"""

    raw_values = requirements.get(field_name, [])
    if not isinstance(raw_values, list) or not all(
        isinstance(value, str) and value.strip() for value in raw_values
    ):
        errors.append(f"data_requirements.{field_name} must be a list of strings")
        return set()
    return {value.strip() for value in raw_values}


def validate_data_requirements(
    records: Iterable[ManifestRecord],
    requirements: Mapping[str, Any] | None,
) -> list[str]:
    """验证训练配置声明的数据规模和覆盖下限。

    许可允许、文件存在并不代表数据足以训练商用品质模型。本门禁按 split 统计
    样本和 ``session_id + scene_id`` 物理组，并要求训练集覆盖指定 Sensor、ISO
    桶和采集模式。配置省略时保持向后兼容，适合已有单元测试和通用库调用。
    """

    if requirements is None:
        return []
    if not isinstance(requirements, Mapping):
        return ["data_requirements must be a mapping"]
    items = list(records)
    errors: list[str] = []
    for field_name, value in (
        ("min_train_records", len([item for item in items if item.split == "train"])),
        ("min_val_records", len([item for item in items if item.split == "val"])),
    ):
        minimum = requirements.get(field_name, 0)
        if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 0:
            errors.append(f"data_requirements.{field_name} must be a non-negative integer")
        elif value < minimum:
            errors.append(f"{field_name}: required >= {minimum}, found {value}")

    for field_name, split in (
        ("min_train_scene_groups", "train"),
        ("min_val_scene_groups", "val"),
    ):
        minimum = requirements.get(field_name, 0)
        if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 0:
            errors.append(f"data_requirements.{field_name} must be a non-negative integer")
            continue
        groups = {
            (item.session_id, item.scene_id) for item in items if item.split == split
        }
        if len(groups) < minimum:
            errors.append(f"{field_name}: required >= {minimum}, found {len(groups)}")

    train_items = [item for item in items if item.split == "train"]
    coverage_fields = (
        ("required_train_sensors", {item.sensor_id for item in train_items}),
        ("required_train_iso_buckets", {item.iso_bucket for item in train_items}),
        ("required_train_modes", {item.mode for item in train_items}),
    )
    for field_name, observed in coverage_fields:
        required = _required_strings(requirements, field_name, errors)
        missing = sorted(required - observed)
        if missing:
            errors.append(f"{field_name}: missing {missing}")
    return errors


def validate_data_policy(
    records: Iterable[ManifestRecord],
    *,
    catalog_path: str | Path,
    purpose: str,
    approval_path: str | Path | None = None,
) -> list[str]:
    """按声明用途检查数据许可；无法证明允许时采用拒绝策略。

    ``commercial_grade`` 表示以商用质量指标研发、但不直接投入生产。
    ``production`` 会额外要求审批文件、证据哈希和目标传感器数据。
    """

    normalized_purpose = purpose.lower()
    if normalized_purpose not in _PURPOSES:
        return [f"unknown training purpose {purpose!r}; expected one of {sorted(_PURPOSES)}"]

    items = list(records)
    catalog = load_yaml(catalog_path)
    datasets, errors = _dataset_index(catalog)
    used_ids = sorted({record.dataset_id for record in items})
    used_entries: dict[str, dict[str, Any]] = {}
    for dataset_id in used_ids:
        entry = datasets.get(dataset_id)
        if entry is None:
            errors.append(f"dataset {dataset_id!r} is absent from the reviewed catalog")
            continue
        used_entries[dataset_id] = entry
        allowed_uses = entry.get("allowed_uses", [])
        if not isinstance(allowed_uses, list) or normalized_purpose not in allowed_uses:
            errors.append(
                f"dataset {dataset_id!r} is not allowed for {normalized_purpose} use"
            )

    # 研发阶段只需要用途被数据目录明确允许；正式生产才进入人工审批门禁。
    if normalized_purpose != "production":
        return errors

    if approval_path is None:
        errors.append("production training requires a data approval file")
        return errors
    approval = load_yaml(approval_path)
    approvals, approval_errors = _approval_index(approval)
    errors.extend(approval_errors)
    for dataset_id in used_ids:
        entry = used_entries.get(dataset_id)
        if entry is not None and str(entry.get("production_default", "")).startswith(
            "prohibited"
        ):
            errors.append(f"dataset {dataset_id!r} is prohibited for production")
        approved = approvals.get(dataset_id)
        if approved is None:
            errors.append(f"dataset {dataset_id!r} has no production approval")
        else:
            errors.extend(_validate_production_approval(dataset_id, approved))

    if not any(
        str(entry.get("role", "")) == "target_sensor" for entry in used_entries.values()
    ):
        errors.append("production training must include an approved target_sensor dataset")
    return errors


def enforce_data_policy(
    records: Iterable[ManifestRecord],
    *,
    catalog_path: str | Path,
    purpose: str,
    context_config: ContextConfig,
    approval_path: str | Path | None = None,
    requirements: Mapping[str, Any] | None = None,
) -> None:
    """组合执行许可、上下文和数据充分性校验，任一失败即阻止训练启动。"""

    items = list(records)
    errors = validate_data_policy(
        items,
        catalog_path=catalog_path,
        purpose=purpose,
        approval_path=approval_path,
    )
    errors.extend(validate_context_coverage(items, context_config))
    errors.extend(validate_data_requirements(items, requirements))
    if errors:
        formatted = "\n".join(f"- {error}" for error in errors)
        raise ValueError(f"training data preflight failed:\n{formatted}")
