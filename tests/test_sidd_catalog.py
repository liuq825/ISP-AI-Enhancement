"""验证官方场景表与主/备镜像批量 URL 的严格逐行绑定。"""

from pathlib import Path

import pytest
import yaml

from isp_ai_enhancement.config import load_yaml
from isp_ai_enhancement.data.sidd import _split_for_scene
from isp_ai_enhancement.data.sidd_catalog import build_sidd_range_config


def _official_fixture() -> tuple[bytes, bytes]:
    """构造两个训练行、一个 held-out 行及十个按五角色排列的 URL。"""

    page = b"""
    <table>
      <tr><td>0001_001_S6_00100_00060_3200_L</td><td>Noisy Raw-RGB</td></tr>
      <tr><td>0009_001_S6_00800_00350_3200_L</td><td>Held for benchmark</td></tr>
      <tr><td>0034_002_GP_00100_00160_3200_N</td><td>Noisy Raw-RGB</td></tr>
    </table>
    """
    urls = [
        f"https://example.test/archive-{index}" for index in range(10)
    ]
    return page, ("\n".join(urls) + "\n").encode()


def test_build_sidd_range_config_binds_five_url_groups_and_excludes_heldout(
    tmp_path: Path,
) -> None:
    """生成器应按训练行绑定 noisy/GT，并保留双帧和来源哈希。"""

    page, mirror = _official_fixture()
    fallback = mirror.replace(b"https://example.test", b"http://fallback.test")
    contents = {
        "https://example.test/scenes": page,
        "https://example.test/urls": mirror,
        "https://example.test/fallback-urls": fallback,
    }

    def fetcher(url: str) -> bytes:
        """从内存映射返回测试来源，不访问公网。"""

        return contents[url]

    output = build_sidd_range_config(
        output=tmp_path / "range.yaml",
        frame_indices=(10, 20),
        source_page="https://example.test/scenes",
        mirror_list="https://example.test/urls",
        fallback_mirror_list="https://example.test/fallback-urls",
        expected_training_scenes=2,
        fetcher=fetcher,
    )
    config = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert config["held_out_scene_count"] == 1
    assert config["frame_indices"] == [10, 20]
    assert [scene["scene"] for scene in config["scenes"]] == [
        "0001_001_S6_00100_00060_3200_L",
        "0034_002_GP_00100_00160_3200_N",
    ]
    assert config["scenes"][0]["noisy_url"].endswith("archive-0")
    assert config["scenes"][0]["ground_truth_url"].endswith("archive-1")
    assert config["scenes"][1]["noisy_url"].endswith("archive-5")
    assert config["scenes"][1]["ground_truth_url"].endswith("archive-6")
    assert config["scenes"][0]["fallback_noisy_url"].startswith(
        "http://fallback.test/"
    )
    assert config["scenes"][0]["fallback_ground_truth_url"].endswith("archive-1")
    assert len(config["source_page_sha256"]) == 64
    assert len(config["mirror_list_sha256"]) == 64
    assert len(config["fallback_mirror_list_sha256"]) == 64


def test_build_sidd_range_config_rejects_shifted_or_incomplete_url_list(
    tmp_path: Path,
) -> None:
    """少一个角色 URL 时必须失败，不能把下一场景 URL 静默前移。"""

    page, mirror = _official_fixture()

    def fetcher(url: str) -> bytes:
        """对 URL 清单故意删除末项，模拟上游格式漂移。"""

        return page if url.endswith("scenes") else b"\n".join(mirror.splitlines()[:-1])

    with pytest.raises(ValueError, match="URL 数"):
        build_sidd_range_config(
            output=tmp_path / "range.yaml",
            source_page="https://example.test/scenes",
            mirror_list="https://example.test/urls",
            fallback_mirror_list=None,
            expected_training_scenes=2,
            fetcher=fetcher,
        )


def test_versioned_medium_config_and_training_thresholds_are_achievable() -> None:
    """仓库 320 对配置应完整覆盖域，且正式门槛不得高于稳定切分产量。"""

    acquisition = load_yaml("resources/sidd_medium_range.yaml")
    scenes = acquisition["scenes"]
    assert len(scenes) == 160
    assert acquisition["held_out_scene_count"] == 40
    assert acquisition["frame_indices"] == [10, 20]
    assert len(acquisition["fallback_mirror_list_sha256"]) == 64
    assert all(
        row["fallback_noisy_url"].startswith("http://130.63.97.225/")
        and row["fallback_ground_truth_url"].startswith("http://130.63.97.225/")
        for row in scenes
    )
    assert {row["scene"].split("_")[1] for row in scenes} == {
        f"{value:03d}" for value in range(1, 11)
    }
    assert {row["scene"].split("_")[2] for row in scenes} == {
        "G4",
        "GP",
        "IP",
        "N6",
        "S6",
    }
    source_pairs = {"train": 0, "val": 0, "test": 0}
    train_sensors: set[str] = set()
    train_iso_buckets: set[str] = set()
    for row in scenes:
        fields = row["scene"].split("_")
        scene_id = fields[1]
        split = _split_for_scene(
            scene_id,
            seed=20260726,
            train_ratio=0.8,
            val_ratio=0.1,
        )
        source_pairs[split] += len(acquisition["frame_indices"])
        if split == "train":
            train_sensors.add(f"sidd_{fields[2]}")
            iso = int(fields[3])
            train_iso_buckets.add(
                "low"
                if iso <= 200
                else "medium"
                if iso <= 800
                else "high"
                if iso <= 3200
                else "extreme"
            )
    assert source_pairs == {"train": 206, "val": 6, "test": 108}

    requirements = load_yaml("configs/train_student_public_baseline.yaml")[
        "data_requirements"
    ]
    assert source_pairs["train"] >= requirements["min_train_source_pairs"]
    assert source_pairs["val"] >= requirements["min_val_source_pairs"]
    assert source_pairs["train"] * 16 >= requirements["min_train_records"]
    assert source_pairs["val"] * 16 >= requirements["min_val_records"]
    assert train_sensors == set(requirements["required_train_sensors"])
    assert train_iso_buckets == set(requirements["required_train_iso_buckets"])
