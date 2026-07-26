"""验证官方 SIDD 元数据、CFA、配对、NLF 与转换流程。"""

from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from isp_ai_enhancement.data.manifest import read_manifest, validate_manifest
from isp_ai_enhancement.data.sidd import (
    SIDD_CFA_PATTERNS,
    SIDDScene,
    discover_sidd_pairs,
    import_sidd_dataset,
    import_sidd_validation_blocks,
    load_sidd_nlf,
    load_sidd_scene_order,
)


def _create_scene(root: Path, name: str, *, mosaic_size: int = 8) -> Path:
    """生成符合 SIDD 命名规则的最小 NumPy 测试场景。"""

    scene = root / name
    scene.mkdir(parents=True)
    mosaic = np.arange(mosaic_size**2, dtype=np.float32).reshape(
        mosaic_size, mosaic_size
    )
    mosaic /= float(mosaic_size**2)
    np.save(scene / f"{name[:4]}_NOISY_RAW_001.npy", mosaic)
    np.save(scene / f"{name[:4]}_GT_RAW_001.npy", np.clip(mosaic + 0.01, 0, 1))
    return scene


def test_official_sidd_cfa_mapping() -> None:
    """五种官方相机代号必须使用已核验的 CFA 排列。"""

    assert SIDD_CFA_PATTERNS == {
        "GP": "BGGR",
        "IP": "RGGB",
        "S6": "GRBG",
        "N6": "BGGR",
        "G4": "BGGR",
    }


def test_parse_scene_and_discover_pairs(tmp_path: Path) -> None:
    """场景目录元数据和噪声/真值文件应被正确解析配对。"""

    name = "0052_002_S6_01600_01000_5500_N"
    scene_dir = _create_scene(tmp_path, name)
    scene = SIDDScene.from_directory(scene_dir)
    assert scene.camera_id == "S6"
    assert scene.iso == 1600
    assert len(discover_sidd_pairs(scene_dir)) == 1


def test_import_sidd_packs_and_preserves_scene_split(tmp_path: Path) -> None:
    """同一场景号应保持相同划分，输出需为四通道 packed RAW。"""

    source = tmp_path / "source"
    first = "0051_002_S6_00100_00060_5500_N"
    second = "0052_002_S6_01600_01000_5500_N"
    _create_scene(source, first)
    _create_scene(source, second)
    nlf = tmp_path / "nlf.csv"
    nlf.write_text(
        "scene_instance_id,beta1_r,beta2_r,beta1_g,beta2_g,beta1_b,beta2_b\n"
        f"{first},0.001,0.00001,0.001,0.00001,0.001,0.00001\n"
        f"{second},0.002,0.00002,0.002,0.00002,0.002,0.00002\n",
        encoding="utf-8",
    )
    manifest = import_sidd_dataset(source, tmp_path / "converted", nlf_csv=nlf)
    records = read_manifest(manifest)
    assert len(records) == 2
    assert records[0].split == records[1].split
    assert records[0].metadata["noise_sigma"] > 0
    assert validate_manifest(records, root=manifest.parent) == []
    with np.load(manifest.parent / records[0].input_path) as archive:
        assert archive["raw"].shape == (4, 4, 4)


def test_import_sidd_rejects_official_held_out_scene(tmp_path: Path) -> None:
    """训练源目录若混入官方 benchmark 场景，导入必须在写产物前失败。"""

    source = tmp_path / "source"
    leaked_name = "0009_001_S6_00800_00350_3200_L"
    _create_scene(source, leaked_name)
    scene_order = tmp_path / "held_out.yaml"
    scene_order.write_text(
        "source_url: https://example.test\n"
        "scenes:\n"
        f"  - {leaked_name}\n",
        encoding="utf-8",
    )
    output = tmp_path / "converted"

    with pytest.raises(ValueError, match="held-out benchmark"):
        import_sidd_dataset(source, output, held_out_scenes=scene_order)
    assert not output.exists()


def test_import_sidd_deterministically_extracts_patches_with_source_identity(
    tmp_path: Path,
) -> None:
    """patch 导入应可复现坐标，并保留独立源配对身份供充分性门禁使用。"""

    source = tmp_path / "source"
    name = "0052_002_S6_01600_01000_5500_N"
    _create_scene(source, name, mosaic_size=64)
    first_manifest = import_sidd_dataset(
        source,
        tmp_path / "first",
        patch_size=16,
        patches_per_pair=3,
        patch_seed=123,
    )
    second_manifest = import_sidd_dataset(
        source,
        tmp_path / "second",
        patch_size=16,
        patches_per_pair=3,
        patch_seed=123,
    )
    first = read_manifest(first_manifest)
    second = read_manifest(second_manifest)
    assert len(first) == len(second) == 3
    assert [item.sample_id for item in first] == [
        "sidd_0052_001_p001",
        "sidd_0052_001_p002",
        "sidd_0052_001_p003",
    ]
    assert {item.metadata["source_pair_id"] for item in first} == {"sidd_0052_001"}
    coordinates = [
        (item.metadata["patch_top"], item.metadata["patch_left"]) for item in first
    ]
    assert len(set(coordinates)) == 3
    assert coordinates == [
        (item.metadata["patch_top"], item.metadata["patch_left"]) for item in second
    ]
    with np.load(first_manifest.parent / first[0].input_path) as archive:
        assert archive["raw"].shape == (4, 16, 16)
    first_npz = first_manifest.parent / first[0].input_path
    original_mtime = first_npz.stat().st_mtime_ns
    repeated_manifest = import_sidd_dataset(
        source,
        tmp_path / "first",
        patch_size=16,
        patches_per_pair=3,
        patch_seed=123,
    )
    assert repeated_manifest == first_manifest
    assert first_npz.stat().st_mtime_ns == original_mtime


def test_nlf_rejects_wrong_header(tmp_path: Path) -> None:
    """缺失官方 NLF 六个系数字段的 CSV 必须被拒绝。"""

    path = tmp_path / "bad.csv"
    path.write_text("scene_instance_id,beta1_r\nx,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid SIDD NLF"):
        load_sidd_nlf(path)


def test_import_sidd_validation_blocks_binds_versioned_scene_cfa(
    tmp_path: Path,
) -> None:
    """验证块第一维应绑定版本化场景顺序，并按各相机 CFA 打包。"""

    scene_order = tmp_path / "scenes.yaml"
    scene_order.write_text(
        "source_url: https://example.test\n"
        "scenes:\n"
        "  - 0009_001_S6_00800_00350_3200_L\n"
        "  - 0021_001_GP_10000_05000_5500_N\n",
        encoding="utf-8",
    )
    noisy = np.arange(2 * 2 * 8 * 8, dtype=np.float32).reshape(2, 2, 8, 8)
    noisy /= float(noisy.max())
    target = np.clip(noisy + 0.01, 0.0, 1.0)
    noisy_mat = tmp_path / "noisy.mat"
    target_mat = tmp_path / "target.mat"
    savemat(noisy_mat, {"ValidationNoisyBlocksRaw": noisy})
    savemat(target_mat, {"ValidationGtBlocksRaw": target})

    scenes = load_sidd_scene_order(scene_order)
    assert [scene.camera_id for scene in scenes] == ["S6", "GP"]
    manifest = import_sidd_validation_blocks(
        noisy_mat,
        target_mat,
        tmp_path / "converted",
        scene_order=scene_order,
    )
    records = read_manifest(manifest)
    assert len(records) == 4
    assert {record.split for record in records} == {"test"}
    assert records[0].metadata["cfa_pattern"] == "GRBG"
    assert records[2].metadata["cfa_pattern"] == "BGGR"
    assert validate_manifest(records, root=manifest.parent) == []
    with np.load(manifest.parent / records[0].input_path) as archive:
        assert archive["raw"].shape == (4, 4, 4)


def test_versioned_official_validation_scene_order_is_complete() -> None:
    """仓库固化的官方 held-out 顺序应恰含 40 场景和五款相机。"""

    scenes = load_sidd_scene_order("resources/sidd_validation_scenes.yaml")
    assert len(scenes) == 40
    assert {scene.camera_id for scene in scenes} == set(SIDD_CFA_PATTERNS)
