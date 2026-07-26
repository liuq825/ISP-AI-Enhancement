"""训练数据清单的数据结构、读写与防泄漏校验。

清单采用一行一个 JSON 对象的 JSONL 格式，便于增量生成、代码审查和流式读取。
``session_id + scene_id`` 被视为不可跨训练/验证/测试划分的最小分组。
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ManifestRecord:
    """描述一对 RAW 输入/标签及其数据治理和分组元数据。"""

    sample_id: str
    dataset_id: str
    input_path: str
    target_path: str
    split: str
    sensor_id: str
    mode: str
    session_id: str
    scene_id: str
    iso_bucket: str
    metadata: dict[str, object]

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> ManifestRecord:
        """从字典构建记录，并在入口处检查所有不可缺少的字段。"""

        required = {
            "sample_id",
            "dataset_id",
            "input_path",
            "target_path",
            "split",
            "sensor_id",
            "mode",
            "session_id",
            "scene_id",
            "iso_bucket",
        }
        missing = sorted(required - value.keys())
        if missing:
            raise ValueError(f"manifest record is missing: {missing}")
        return cls(
            sample_id=str(value["sample_id"]),
            dataset_id=str(value["dataset_id"]),
            input_path=str(value["input_path"]),
            target_path=str(value["target_path"]),
            split=str(value["split"]),
            sensor_id=str(value["sensor_id"]),
            mode=str(value["mode"]),
            session_id=str(value["session_id"]),
            scene_id=str(value["scene_id"]),
            iso_bucket=str(value["iso_bucket"]),
            metadata=dict(value.get("metadata", {})),
        )

    def as_dict(self) -> dict[str, object]:
        """转换为可直接写入 JSONL 的普通字典。"""

        return {
            "sample_id": self.sample_id,
            "dataset_id": self.dataset_id,
            "input_path": self.input_path,
            "target_path": self.target_path,
            "split": self.split,
            "sensor_id": self.sensor_id,
            "mode": self.mode,
            "session_id": self.session_id,
            "scene_id": self.scene_id,
            "iso_bucket": self.iso_bucket,
            "metadata": self.metadata,
        }


def read_manifest(path: str | Path) -> list[ManifestRecord]:
    """读取 UTF-8 JSONL 清单，并在错误信息中保留准确行号。"""

    source = Path(path)
    records: list[ManifestRecord] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("record is not a JSON object")
                records.append(ManifestRecord.from_dict(value))
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(f"{source}:{line_number}: {error}") from error
    return records


def write_manifest(records: Iterable[ManifestRecord], path: str | Path) -> None:
    """以稳定键序和 LF 换行写出清单，便于跨平台比较与版本控制。"""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.as_dict(), ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def validate_manifest(
    records: Iterable[ManifestRecord],
    *,
    root: str | Path | None = None,
    require_files: bool = True,
) -> list[str]:
    """检查 ID、划分、文件存在性以及场景级数据泄漏。

    返回全部错误而不是遇到第一项就退出，以便数据准备人员一次完成修复。
    ``golden`` 是独立发布门禁集，可与常规划分并存但不参与泄漏判定。
    """

    items = list(records)
    errors: list[str] = []
    seen_ids: set[str] = set()
    group_splits: dict[tuple[str, str], set[str]] = {}
    base = Path(root) if root is not None else None
    valid_splits = {"train", "val", "test", "golden"}
    for record in items:
        if record.sample_id in seen_ids:
            errors.append(f"duplicate sample_id: {record.sample_id}")
        seen_ids.add(record.sample_id)
        if not record.dataset_id.strip():
            errors.append(f"{record.sample_id}: dataset_id must not be empty")
        if record.split not in valid_splits:
            errors.append(f"{record.sample_id}: invalid split {record.split!r}")
        # 同一拍摄会话中的同一场景必须整体分到一个集合，不能按 patch 随机拆分。
        group = (record.session_id, record.scene_id)
        group_splits.setdefault(group, set()).add(record.split)
        if require_files:
            for field_name, raw_path in (
                ("input_path", record.input_path),
                ("target_path", record.target_path),
            ):
                candidate = Path(raw_path)
                if not candidate.is_absolute() and base is not None:
                    candidate = base / candidate
                if not candidate.is_file():
                    errors.append(f"{record.sample_id}: missing {field_name} {candidate}")
    for group, splits in group_splits.items():
        non_golden = splits - {"golden"}
        if len(non_golden) > 1:
            errors.append(f"session/scene leakage for {group}: appears in {sorted(non_golden)}")
    if not items:
        errors.append("manifest contains no records")
    return errors
