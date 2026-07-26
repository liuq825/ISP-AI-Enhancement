"""验证远程 ZIP 单成员提取、断点复用和 held-out 隔离。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from isp_ai_enhancement import cli
from isp_ai_enhancement.data.sidd_remote import (
    fetch_sidd_raw_pair,
    fetch_sidd_raw_subset,
    sidd_subset_status,
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
    progress: list[str] = []

    def factory(url: str) -> ZipFile:
        """记录被打开的 URL，并返回对应本地测试 ZIP。"""

        opened.append(url)
        return ZipFile(archives[url])

    receipt_path = fetch_sidd_raw_subset(
        config=config,
        output_dir=tmp_path / "output",
        held_out_scenes=held_out,
        archive_factory=factory,
        progress_callback=progress.append,
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["scene_count"] == 1
    assert receipt["pairs"][0]["receipt_sha256"]
    assert opened == [
        "https://example.test/noisy.zip",
        "https://example.test/target.zip",
    ]
    status = sidd_subset_status(config=config, output_dir=tmp_path / "output")
    assert status["status"] == "complete"
    assert status["completed_scenes"] == 1
    assert status["completed_pairs"] == 1
    assert status["completion_percent"] == 100.0
    assert progress == [
        f"[1/1] 开始获取 {scene} frames [010]",
        f"[1/1] 已校验 {scene} frames [010]",
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


def test_fetch_sidd_subset_cli_separates_progress_from_result(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """CLI 应把进度写入 stderr，stdout 只输出最终机器可解析的收据路径。"""

    def fake_fetch(**arguments: object) -> Path:
        """模拟批量获取并主动触发一个进度事件。"""

        callback = arguments["progress_callback"]
        assert callable(callback)
        callback("[1/1] 已校验测试场景")
        return Path("datasets/SIDD_Training_Subset/subset.receipt.json")

    monkeypatch.setattr(cli, "fetch_sidd_raw_subset", fake_fetch)
    arguments = argparse.Namespace(
        config="resources/sidd_training_subset.yaml",
        output="datasets/SIDD_Training_Subset",
        held_out_scenes="resources/sidd_validation_scenes.yaml",
        max_member_bytes=512 * 1024 * 1024,
        progress_file=str(tmp_path / "progress.log"),
        max_attempts=4,
        retry_backoff_seconds=5.0,
    )
    assert cli._fetch_sidd_subset(arguments) == 0
    captured = capsys.readouterr()
    expected = Path("datasets/SIDD_Training_Subset/subset.receipt.json")
    assert captured.out == f"{expected}\n"
    assert captured.err == "[1/1] 已校验测试场景\n"
    assert (tmp_path / "progress.log").read_text(encoding="utf-8") == captured.err


def test_fetch_sidd_subset_supports_multiple_frames_per_scene(tmp_path: Path) -> None:
    """双帧配置应生成两对文件，并在集合收据区分场景数与配对数。"""

    scene = "0001_001_S6_00100_00060_3200_L"
    noisy_zip = tmp_path / "noisy.zip"
    target_zip = tmp_path / "target.zip"
    with ZipFile(noisy_zip, "w") as archive:
        archive.writestr("raw/0001_NOISY_RAW_010.MAT", b"noisy-10")
        archive.writestr("raw/0001_NOISY_RAW_020.MAT", b"noisy-20")
    with ZipFile(target_zip, "w") as archive:
        archive.writestr("raw/0001_GT_RAW_010.MAT", b"target-10")
        archive.writestr("raw/0001_GT_RAW_020.MAT", b"target-20")
    held_out = tmp_path / "held_out.yaml"
    held_out.write_text(
        "source_url: https://example.test\n"
        "scenes:\n"
        "  - 0009_001_S6_00800_00350_3200_L\n",
        encoding="utf-8",
    )
    config = tmp_path / "subset.yaml"
    config.write_text(
        "frame_indices: [10, 20]\n"
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
        """把同一远程归档映射给两个帧请求。"""

        opened.append(url)
        return ZipFile(archives[url])

    receipt_path = fetch_sidd_raw_subset(
        config=config,
        output_dir=tmp_path / "output",
        held_out_scenes=held_out,
        archive_factory=factory,
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["format_version"] == 2
    assert receipt["scene_count"] == 1
    assert receipt["pair_count"] == 2
    assert receipt["frame_indices"] == [10, 20]
    assert [item["frame_index"] for item in receipt["pairs"]] == [10, 20]
    # 两帧共享 noisy/GT 两次归档打开，而不是每对各打开两次。
    assert opened == [
        "https://example.test/noisy.zip",
        "https://example.test/target.zip",
    ]


def test_fetch_sidd_subset_retries_network_error_but_not_content_error(
    tmp_path: Path,
) -> None:
    """临时 I/O 错误应重试，确定性成员错配必须立即失败。"""

    scene = "0001_001_S6_00100_00060_3200_L"
    noisy_zip = tmp_path / "noisy.zip"
    target_zip = tmp_path / "target.zip"
    _create_zip(noisy_zip, "0001_NOISY_RAW_010.MAT", b"noisy")
    _create_zip(target_zip, "0001_GT_RAW_010.MAT", b"target")
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
    calls = 0
    progress: list[str] = []

    def flaky_factory(url: str) -> ZipFile:
        """首次打开模拟连接重置，随后返回正确本地归档。"""

        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("temporary reset")
        return ZipFile(noisy_zip if "noisy" in url else target_zip)

    fetch_sidd_raw_subset(
        config=config,
        output_dir=tmp_path / "retry-output",
        held_out_scenes=held_out,
        archive_factory=flaky_factory,
        progress_callback=progress.append,
        max_attempts=2,
        retry_backoff_seconds=0,
    )
    assert calls == 3
    assert any("OSError" in message and "2/2" in message for message in progress)

    bad_zip = tmp_path / "bad.zip"
    _create_zip(bad_zip, "wrong-member.MAT", b"wrong")
    bad_calls = 0

    def mismatched_factory(_url: str) -> ZipFile:
        """始终返回成员错配归档并记录调用次数。"""

        nonlocal bad_calls
        bad_calls += 1
        return ZipFile(bad_zip)

    with pytest.raises(ValueError, match="应唯一包含"):
        fetch_sidd_raw_subset(
            config=config,
            output_dir=tmp_path / "bad-output",
            held_out_scenes=held_out,
            archive_factory=mismatched_factory,
            max_attempts=4,
            retry_backoff_seconds=0,
        )
    assert bad_calls == 1
