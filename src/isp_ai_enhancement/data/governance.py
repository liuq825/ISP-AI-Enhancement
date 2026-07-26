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
    """Require a versioned registry entry for every Sensor used by training."""
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


def validate_data_policy(
    records: Iterable[ManifestRecord],
    *,
    catalog_path: str | Path,
    purpose: str,
    approval_path: str | Path | None = None,
) -> list[str]:
    """Fail closed when a manifest is not licensed for the declared training purpose."""
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
) -> None:
    items = list(records)
    errors = validate_data_policy(
        items,
        catalog_path=catalog_path,
        purpose=purpose,
        approval_path=approval_path,
    )
    errors.extend(validate_context_coverage(items, context_config))
    if errors:
        formatted = "\n".join(f"- {error}" for error in errors)
        raise ValueError(f"training data preflight failed:\n{formatted}")
