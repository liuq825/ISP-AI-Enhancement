"""官方 SIDD 场景级 RAW 数据发现、校验与项目格式转换。

SIDD 的 ``*_NOISY_RAW``/``*_GT_RAW`` 文件是二维 Bayer 马赛克，而项目网络
接收统一顺序的四通道 packed RAW。本模块根据官方相机 CFA 表完成打包，
保留源文件 SHA256、NLF、ISO 和色温等来源信息，并按场景划分数据集。
"""

from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .context import canonical_pack_bayer
from .manifest import ManifestRecord, write_manifest

SIDD_CFA_PATTERNS: dict[str, str] = {
    "GP": "BGGR",
    "IP": "RGGB",
    "S6": "GRBG",
    "N6": "BGGR",
    "G4": "BGGR",
}

_SCENE_PATTERN = re.compile(
    r"^(?P<instance>\d{4})_(?P<scene>\d{3})_(?P<camera>GP|IP|S6|N6|G4)_"
    r"(?P<iso>\d{5})_(?P<shutter>\d{5})_(?P<cct>\d{4})_(?P<brightness>[LNH])$",
    re.IGNORECASE,
)
_RAW_SUFFIXES = {".mat", ".npy"}


@dataclass(frozen=True)
class SIDDScene:
    """从 SIDD 官方目录名解析出的拍摄场景元数据。"""

    instance_id: str
    scene_id: str
    camera_id: str
    iso: int
    shutter_denominator: int
    cct: int
    brightness: str

    @classmethod
    def from_directory(cls, directory: Path) -> SIDDScene:
        """严格解析官方目录命名；格式不符时拒绝猜测元数据。"""

        match = _SCENE_PATTERN.fullmatch(directory.name)
        if match is None:
            raise ValueError(f"invalid SIDD scene directory name: {directory.name}")
        values = match.groupdict()
        return cls(
            instance_id=values["instance"],
            scene_id=values["scene"],
            camera_id=values["camera"].upper(),
            iso=int(values["iso"]),
            shutter_denominator=int(values["shutter"]),
            cct=int(values["cct"]),
            brightness=values["brightness"].upper(),
        )


def _sha256(path: Path) -> str:
    """以分块方式计算源文件 SHA256，避免大 RAW 文件整体进入内存。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_pair_key(path: Path, role: str) -> str:
    """移除 NOISY/GT 角色差异，得到可用于配对的稳定文件键。"""

    marker = f"_{role}_RAW"
    stem = path.stem.upper()
    if marker not in stem:
        raise ValueError(f"{path} does not contain {marker}")
    return stem.replace(marker, "_RAW", 1)


def discover_sidd_pairs(scene_dir: str | Path) -> list[tuple[Path, Path]]:
    """递归发现一个场景中的噪声/真值 RAW，并拒绝任何不完整配对。"""

    directory = Path(scene_dir)
    noisy: dict[str, Path] = {}
    target: dict[str, Path] = {}
    for path in directory.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in _RAW_SUFFIXES:
            continue
        upper = path.stem.upper()
        if "_NOISY_RAW" in upper:
            noisy[_raw_pair_key(path, "NOISY")] = path
        elif "_GT_RAW" in upper:
            target[_raw_pair_key(path, "GT")] = path
    missing_target = sorted(set(noisy) - set(target))
    missing_noisy = sorted(set(target) - set(noisy))
    if missing_target or missing_noisy:
        raise ValueError(
            f"{directory}: unpaired SIDD files; "
            f"missing GT={missing_target}, missing noisy={missing_noisy}"
        )
    if not noisy:
        raise ValueError(f"{directory}: no SIDD NOISY_RAW/GT_RAW pairs found")
    return [(noisy[key], target[key]) for key in sorted(noisy)]


def _select_mat_array(values: dict[str, object], path: Path) -> np.ndarray:
    """从 MAT 命名空间中选择唯一二维数值 RAW 数组。"""

    candidates: list[tuple[str, np.ndarray]] = []
    for key, value in values.items():
        if key.startswith("__"):
            continue
        array = np.asarray(value)
        if array.ndim == 2 and np.issubdtype(array.dtype, np.number):
            candidates.append((key, array))
    if len(candidates) != 1:
        names = [key for key, _array in candidates]
        raise ValueError(f"{path}: expected one numeric 2D RAW array, found {names}")
    return candidates[0][1]


def _load_mat(path: Path) -> np.ndarray:
    """兼容传统 MAT 与 MATLAB v7.3/HDF5 两种 SIDD 文件格式。"""

    try:
        from scipy.io import loadmat

        return _select_mat_array(loadmat(path), path)
    except NotImplementedError:
        pass
    except ImportError as error:
        raise RuntimeError(
            "reading SIDD .MAT files requires the 'data' extra: "
            "pip install -e '.[data]'"
        ) from error

    try:
        import h5py
    except ImportError as error:
        raise RuntimeError(
            "MATLAB v7.3 files require h5py from the 'data' extra"
        ) from error
    with h5py.File(path, "r") as archive:
        values = {
            key: np.asarray(value).T
            for key, value in archive.items()
            if hasattr(value, "shape")
        }
    return _select_mat_array(values, path)


def load_sidd_raw(path: str | Path) -> np.ndarray:
    """读取 SIDD RAW 并校验二维偶数尺寸、有限值及 ``[0, 1]`` 范围。"""

    source = Path(path)
    if source.suffix.lower() == ".npy":
        raw = np.load(source, allow_pickle=False)
    else:
        raw = _load_mat(source)
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[0] % 2 or raw.shape[1] % 2:
        raise ValueError(f"{source}: RAW mosaic must be an even-sized 2D array")
    if not np.isfinite(raw).all():
        raise ValueError(f"{source}: RAW contains NaN or infinity")
    if float(raw.min()) < -1e-6 or float(raw.max()) > 1.0 + 1e-6:
        raise ValueError(f"{source}: SIDD RAW must be normalized to [0, 1]")
    return np.clip(raw, 0.0, 1.0)


def load_sidd_nlf(path: str | Path) -> dict[str, tuple[float, ...]]:
    """读取官方噪声水平函数 CSV，按场景实例名建立六参数索引。"""

    result: dict[str, tuple[float, ...]] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = (
            "scene_instance_id",
            "beta1_r",
            "beta2_r",
            "beta1_g",
            "beta2_g",
            "beta1_b",
            "beta2_b",
        )
        if reader.fieldnames is None or any(name not in reader.fieldnames for name in required):
            raise ValueError(f"{path}: invalid SIDD NLF CSV header")
        for row in reader:
            scene_id = str(row["scene_instance_id"])
            result[scene_id] = tuple(float(row[name]) for name in required[1:])
    return result


def _noise_sigma(nlf: tuple[float, ...] | None, reference_level: float = 0.18) -> float:
    """在参考灰度处把 RGB 三组 NLF 系数折算为单一噪声强度条件。"""

    if nlf is None:
        return 0.0
    variances = [
        nlf[index] * reference_level + nlf[index + 1] for index in (0, 2, 4)
    ]
    return float(np.sqrt(max(0.0, sum(variances) / len(variances))))


def _split_for_scene(
    scene_id: str,
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> str:
    """对场景 ID 做稳定哈希划分，杜绝同场景跨集合泄漏。"""

    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("split ratios must satisfy train>0, val>=0, and train+val<1")
    digest = hashlib.sha256(f"{seed}:{scene_id}".encode()).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2**64
    if fraction < train_ratio:
        return "train"
    if fraction < train_ratio + val_ratio:
        return "val"
    return "test"


def _iso_bucket(iso: int) -> str:
    """把 ISO 映射到低、中、高、极高四个质量统计桶。"""

    if iso <= 200:
        return "low"
    if iso <= 800:
        return "medium"
    if iso <= 3200:
        return "high"
    return "extreme"


def import_sidd_dataset(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    nlf_csv: str | Path | None = None,
    split_seed: int = 20260726,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Path:
    """把解压后的官方 SIDD RAW 树转换为项目统一的 packed RAW 数据集。

    每一对源文件都会记录 SHA256；分组键使用场景 ID，确保同一场景的多个
    噪声帧不会被拆到训练和测试集合。输出 NPZ 的通道顺序恒为
    ``[R, Gr, Gb, B]``，与手机原始 CFA 类型无关。
    """

    source = Path(source_dir)
    output = Path(output_dir)
    sample_dir = output / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    nlf_values = load_sidd_nlf(nlf_csv) if nlf_csv is not None else {}
    scene_dirs = sorted(
        path
        for path in source.rglob("*")
        if path.is_dir() and _SCENE_PATTERN.fullmatch(path.name)
    )
    if not scene_dirs:
        raise ValueError(f"{source}: no SIDD scene directories found")

    records: list[ManifestRecord] = []
    for scene_dir in scene_dirs:
        scene = SIDDScene.from_directory(scene_dir)
        cfa = SIDD_CFA_PATTERNS[scene.camera_id]
        nlf = nlf_values.get(scene_dir.name)
        # 只对场景 ID 做一次划分；场景内所有噪声/真值对继承同一 split。
        split = _split_for_scene(
            scene.scene_id,
            seed=split_seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )
        pairs = discover_sidd_pairs(scene_dir)
        for pair_index, (noisy_path, target_path) in enumerate(pairs, start=1):
            noisy_mosaic = load_sidd_raw(noisy_path)
            target_mosaic = load_sidd_raw(target_path)
            if noisy_mosaic.shape != target_mosaic.shape:
                raise ValueError(
                    f"{scene_dir}: noisy/GT shape mismatch "
                    f"{noisy_mosaic.shape} != {target_mosaic.shape}"
                )
            noisy = canonical_pack_bayer(torch.from_numpy(noisy_mosaic), cfa).numpy()
            target = canonical_pack_bayer(torch.from_numpy(target_mosaic), cfa).numpy()
            sample_id = f"sidd_{scene.instance_id}_{pair_index:03d}"
            converted_input = sample_dir / f"{sample_id}_input.npz"
            converted_target = sample_dir / f"{sample_id}_target.npz"
            np.savez_compressed(converted_input, raw=noisy)
            np.savez_compressed(converted_target, raw=target)
            records.append(
                ManifestRecord(
                    sample_id=sample_id,
                    dataset_id="sidd",
                    input_path=converted_input.relative_to(output).as_posix(),
                    target_path=converted_target.relative_to(output).as_posix(),
                    split=split,
                    sensor_id=f"sidd_{scene.camera_id}",
                    mode="single",
                    session_id="sidd",
                    scene_id=scene.scene_id,
                    iso_bucket=_iso_bucket(scene.iso),
                    metadata={
                        "noise_sigma": _noise_sigma(nlf),
                        "exposure_ratio": 1.0,
                        "wb_rg": 1.0,
                        "wb_bg": 1.0,
                        "cfa_pattern": cfa,
                        "iso": scene.iso,
                        "shutter_denominator": scene.shutter_denominator,
                        "cct": scene.cct,
                        "illuminant_brightness": scene.brightness,
                        "nlf": list(nlf) if nlf is not None else None,
                        "source_input_sha256": _sha256(noisy_path),
                        "source_target_sha256": _sha256(target_path),
                    },
                )
            )
    manifest_path = output / "manifest.jsonl"
    write_manifest(records, manifest_path)
    return manifest_path
