"""验证远程 ZIP 单成员提取、断点复用和 held-out 隔离。"""

from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from isp_ai_enhancement.data.sidd_remote import (
    fetch_sidd_raw_pair,
    fetch_sidd_raw_subset,
)


def _create_zip(path: Path, member: str, payload: bytes) -> None:
    """创建包含目录前缀的最小 ZIP，模拟官方场景归档。"""

    with ZipFile(path, "w") as archive:
        archive.writestr(f"nested/{member}", payload)


def test_fetch_sidd_pair_extracts_members_and_reuses_verified_files(
    tmp_path: Path,
) -> None:
    """提取应忽略 ZIP 内目录，重跑时按大小/CRC 复用完整文件。"""

    scene = "0001_001_S6_00100_00060_3200_L"
    noisy_zip = tmp_path / "noisy.zip"
    target_zip = tmp_path / "target.zip"
    _create_zip(noisy_zip, "0001_NOISY_RAW_010.MAT", b"noisy-mat")
    _create_zip(target_zip, "0001_GT_RAW_010.MAT", b"target-mat")
    scene_order = tmp_path / "held_out.yaml"
    scene_order.write_text(
        "source_url: https://example.test\n"
        "scenes:\n"
        "  - 0009_001_S6_00800_00350_3200_L\n",
        encoding="utf-8",
    )
    archives = {
        "https://example.test/noisy.zip": noisy_zip,
        "https://example.test/target.zip": target_zip,
    }

    def factory(url: str) -> ZipFile:
        """把测试 URL 映射为本地 ZIP，不发起外部网络请求。"""

        return ZipFile(archives[url])

    receipt_path = fetch_sidd_raw_pair(
        scene_name=scene,
        noisy_zip_url="https://example.test/noisy.zip",
        ground_truth_zip_url="https://example.test/target.zip",
        frame_index=10,
        output_dir=tmp_path / "output",
        held_out_scenes=scene_order,
        archive_factory=factory,
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["noisy"]["sha256"]
    assert not receipt["noisy"]["reused_existing"]
    assert (receipt_path.parent / "0001_NOISY_RAW_010.MAT").read_bytes() == b"noisy-mat"
    assert (receipt_path.parent / "0001_GT_RAW_010.MAT").read_bytes() == b"target-mat"

    second = fetch_sidd_raw_pair(
        scene_name=scene,
        noisy_zip_url="https://example.test/noisy.zip",
        ground_truth_zip_url="https://example.test/target.zip",
        frame_index=10,
        output_dir=tmp_path / "output",
        held_out_scenes=scene_order,
        archive_factory=factory,
    )
    repeated = json.loads(second.read_text(encoding="utf-8"))
    assert repeated["noisy"]["reused_existing"]
    assert repeated["ground_truth"]["reused_existing"]


def test_fetch_sidd_pair_rejects_held_out_scene_before_network(tmp_path: Path) -> None:
    """命中 benchmark 场景时必须在打开远程 ZIP 之前拒绝。"""

    held_out = "0009_001_S6_00800_00350_3200_L"
    scene_order = tmp_path / "held_out.yaml"
    scene_order.write_text(
        "source_url: https://example.test\n"
        "scenes:\n"
        f"  - {held_out}\n",
        encoding="utf-8",
    )

    def forbidden_factory(_url: str) -> ZipFile:
        """若隔离检查正确，本工厂永远不应被调用。"""

        raise AssertionError("network factory must not be called")

    with pytest.raises(ValueError, match="held-out benchmark"):
        fetch_sidd_raw_pair(
            scene_name=held_out,
            noisy_zip_url="https://example.test/noisy.zip",
            ground_truth_zip_url="https://example.test/target.zip",
            frame_index=10,
            output_dir=tmp_path / "output",
            held_out_scenes=scene_order,
            archive_factory=forbidden_factory,
        )


def test_fetch_sidd_subset_validates_all_rows_and_writes_collection_receipt(
    tmp_path: Path,
) -> None:
    """批量获取应产生集合收据，并在网络前拒绝后续行的重复场景。"""

    scene = "0001_001_S6_00100_00060_3200_L"
    noisy_zip = tmp_path / "noisy.zip"
    target_zip = tmp_path / "target.zip"
    _create_zip(noisy_zip, "0001_NOISY_RAW_010.MAT", b"noisy-mat")
    _create_zip(target_zip, "0001_GT_RAW_010.MAT", b"target-mat")
    held_out = tmp_path / "held_out.yaml"
    held_out.write_text(
        "source_url: https://example.test\n"
        "scenes:\n"
        "  - 0009_001_S6_00800_00350_3200_L\n",
        encoding="utf-8",
    )
    config = tmp_path / "subset.yaml"
    config.write_text(
        "frame_index: 10\n"
        "scenes:\n"
        f"  - scene: {scene}\n"
        "    noisy_url: https://example.test/noisy.zip\n"
        "    ground_truth_url: https://example.test/target.zip\n",
        encoding="utf-8",
    )
    archives = {
        "https://example.test/noisy.zip": noisy_zip,
        "https://example.test/target.zip": target_zip,
    }
    opened: list[str] = []

    def factory(url: str) -> ZipFile:
        """记录被打开的 URL，并返回对应本地测试 ZIP。"""

        opened.append(url)
        return ZipFile(archives[url])

    receipt_path = fetch_sidd_raw_subset(
        config=config,
        output_dir=tmp_path / "output",
        held_out_scenes=held_out,
        archive_factory=factory,
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["scene_count"] == 1
    assert receipt["scenes"][0]["receipt_sha256"]
    assert opened == [
        "https://example.test/noisy.zip",
        "https://example.test/target.zip",
    ]

    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text(
        "frame_index: 10\n"
        "scenes:\n"
        f"  - scene: {scene}\n"
        "    noisy_url: https://example.test/noisy.zip\n"
        "    ground_truth_url: https://example.test/target.zip\n"
        f"  - scene: {scene}\n"
        "    noisy_url: https://example.test/never-open.zip\n"
        "    ground_truth_url: https://example.test/never-open.zip\n",
        encoding="utf-8",
    )
    opened.clear()
    with pytest.raises(ValueError, match="重复场景"):
        fetch_sidd_raw_subset(
            config=duplicate,
            output_dir=tmp_path / "duplicate-output",
            held_out_scenes=held_out,
            archive_factory=factory,
        )
    assert opened == []
