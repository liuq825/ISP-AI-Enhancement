"""验证官方 SIDD 元数据、CFA、配对、NLF 与转换流程。"""

from pathlib import Path

import numpy as np
import pytest

from isp_ai_enhancement.data.manifest import read_manifest, validate_manifest
from isp_ai_enhancement.data.sidd import (
    SIDD_CFA_PATTERNS,
    SIDDScene,
    discover_sidd_pairs,
    import_sidd_dataset,
    load_sidd_nlf,
)


def _create_scene(root: Path, name: str) -> Path:
    """生成符合 SIDD 命名规则的最小 NumPy 测试场景。"""

    scene = root / name
    scene.mkdir(parents=True)
    mosaic = np.arange(64, dtype=np.float32).reshape(8, 8) / 64.0
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


def test_nlf_rejects_wrong_header(tmp_path: Path) -> None:
    """缺失官方 NLF 六个系数字段的 CSV 必须被拒绝。"""

    path = tmp_path / "bad.csv"
    path.write_text("scene_instance_id,beta1_r\nx,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid SIDD NLF"):
        load_sidd_nlf(path)
