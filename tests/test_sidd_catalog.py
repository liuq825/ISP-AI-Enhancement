"""验证官方场景表与 Mirror 2 批量 URL 的严格逐行绑定。"""

from pathlib import Path

import pytest
import yaml

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
    contents = {
        "https://example.test/scenes": page,
        "https://example.test/urls": mirror,
    }

    def fetcher(url: str) -> bytes:
        """从内存映射返回测试来源，不访问公网。"""

        return contents[url]

    output = build_sidd_range_config(
        output=tmp_path / "range.yaml",
        frame_indices=(10, 20),
        source_page="https://example.test/scenes",
        mirror_list="https://example.test/urls",
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
    assert len(config["source_page_sha256"]) == 64
    assert len(config["mirror_list_sha256"]) == 64


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
            expected_training_scenes=2,
            fetcher=fetcher,
        )
