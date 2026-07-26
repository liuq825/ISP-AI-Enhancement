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

from isp_ai_enhancement.config import load_yaml

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
_VALIDATION_NOISY_VARIABLE = "ValidationNoisyBlocksRaw"
_VALIDATION_GT_VARIABLE = "ValidationGtBlocksRaw"


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


def load_sidd_scene_order(path: str | Path) -> list[SIDDScene]:
    """读取版本化 SIDD 验证场景顺序，并严格拒绝重复或非法场景名。

    官方验证块 MAT 只有 ``40×32×256×256`` 数组，不携带相机/CFA 字段；
    因此第一维与场景表的绑定本身就是数据契约，必须作为仓库资源审计。
    """

    values = load_yaml(path)
    raw_scenes = values.get("scenes")
    if not isinstance(raw_scenes, list) or not raw_scenes:
        raise ValueError(f"{path}: 'scenes' must be a non-empty list")
    scenes = [SIDDScene.from_directory(Path(str(value))) for value in raw_scenes]
    names = [str(value) for value in raw_scenes]
    if len(names) != len(set(names)):
        raise ValueError(f"{path}: scene order contains duplicate names")
    return scenes


def _sidd_mat_shape(path: Path, variable: str) -> tuple[int, ...]:
    """只读 MAT 目录并返回指定变量形状，避免为预检加载数百 MB 数组。"""

    try:
        from scipy.io import whosmat
    except ImportError as error:
        raise RuntimeError(
            "reading SIDD validation blocks requires: pip install -e '.[data]'"
        ) from error
    matches = [shape for name, shape, _dtype in whosmat(path) if name == variable]
    if len(matches) != 1:
        raise ValueError(f"{path}: expected exactly one MAT variable {variable!r}")
    return tuple(int(value) for value in matches[0])


def _load_sidd_block_array(path: Path, variable: str) -> np.ndarray:
    """加载一个 SIDD 验证块变量并执行形状、有限值和归一化范围校验。"""

    try:
        from scipy.io import loadmat
    except ImportError as error:
        raise RuntimeError(
            "reading SIDD validation blocks requires: pip install -e '.[data]'"
        ) from error
    values = loadmat(path, variable_names=[variable])
    if variable not in values:
        raise ValueError(f"{path}: MAT variable {variable!r} is missing")
    blocks = np.asarray(values[variable], dtype=np.float32)
    if blocks.ndim != 4 or blocks.shape[1] <= 0:
        raise ValueError(
            f"{path}: {variable} must have shape images×blocks×H×W, got {blocks.shape}"
        )
    if blocks.shape[-2] % 2 or blocks.shape[-1] % 2:
        raise ValueError(f"{path}: validation block dimensions must be even")
    if not np.isfinite(blocks).all():
        raise ValueError(f"{path}: validation blocks contain NaN or infinity")
    if float(blocks.min()) < -1e-6 or float(blocks.max()) > 1.0 + 1e-6:
        raise ValueError(f"{path}: validation blocks must be normalized to [0, 1]")
    return np.clip(blocks, 0.0, 1.0)


def _save_raw_npz(path: Path, raw: np.ndarray) -> None:
    """原子写入单个 packed RAW NPZ，避免中断后留下貌似完整的文件。"""

    temporary = path.with_name(f"{path.name}.tmp.npz")
    np.savez_compressed(temporary, raw=np.asarray(raw, dtype=np.float32))
    temporary.replace(path)


def _write_sidd_block_role(
    blocks: np.ndarray,
    scenes: list[SIDDScene],
    sample_dir: Path,
    role: str,
) -> None:
    """按每个场景的 CFA 打包验证块，并写成输入或目标 NPZ 文件。"""

    if blocks.shape[0] != len(scenes):
        raise ValueError(
            f"validation block image count {blocks.shape[0]} "
            f"does not match scene order {len(scenes)}"
        )
    for scene_index, scene in enumerate(scenes):
        cfa = SIDD_CFA_PATTERNS[scene.camera_id]
        # 一次打包同场景的全部 block，避免 1,280 次单独创建 Torch 张量。
        packed = canonical_pack_bayer(torch.from_numpy(blocks[scene_index]), cfa).numpy()
        for block_index, raw in enumerate(packed, start=1):
            sample_id = f"sidd_val_{scene.instance_id}_{block_index:02d}"
            _save_raw_npz(sample_dir / f"{sample_id}_{role}.npz", raw)


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


def _patch_coordinates(
    *,
    height: int,
    width: int,
    patch_size: int,
    patch_count: int,
    seed_token: str,
) -> list[tuple[int, int]]:
    """从 packed RAW 平面确定性抽取不重复 patch 左上角坐标。"""

    if patch_size <= 0 or patch_size % 16:
        raise ValueError("patch_size 必须为 16 的正整数倍")
    if patch_count <= 0:
        raise ValueError("patches_per_pair 必须为正整数")
    if height < patch_size or width < patch_size:
        raise ValueError(
            f"packed RAW {height}×{width} 小于 patch_size={patch_size}"
        )
    possible = (height - patch_size + 1) * (width - patch_size + 1)
    if patch_count > possible:
        raise ValueError(f"patches_per_pair={patch_count} 超过可用位置 {possible}")
    seed = int.from_bytes(hashlib.sha256(seed_token.encode()).digest()[:8], "big")
    generator = np.random.default_rng(seed)
    coordinates: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    while len(coordinates) < patch_count:
        coordinate = (
            int(generator.integers(0, height - patch_size + 1)),
            int(generator.integers(0, width - patch_size + 1)),
        )
        if coordinate not in seen:
            seen.add(coordinate)
            coordinates.append(coordinate)
    return coordinates


def import_sidd_dataset(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    nlf_csv: str | Path | None = None,
    held_out_scenes: str | Path | None = None,
    split_seed: int = 20260726,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    patch_size: int | None = None,
    patches_per_pair: int = 1,
    patch_seed: int = 20260727,
) -> Path:
    """把解压后的官方 SIDD RAW 树转换为项目统一的 packed RAW 数据集。

    每一对源文件都会记录 SHA256；分组键使用场景 ID，确保同一场景的多个
    噪声帧不会被拆到训练和测试集合。若提供 ``held_out_scenes``，任何官方
    benchmark 场景一旦出现在训练源目录中都会被硬拒绝。输出 NPZ 的通道顺序
    恒为 ``[R, Gr, Gb, B]``，与手机原始 CFA 类型无关。提供 ``patch_size`` 时，
    每个源配对按固定 seed 生成不重复 packed RAW patch，并保留 ``source_pair_id``，
    供数据充分性门禁区分独立拍摄与派生裁剪。
    """

    if patch_size is None and patches_per_pair != 1:
        raise ValueError("未设置 patch_size 时 patches_per_pair 必须为 1")
    source = Path(source_dir)
    output = Path(output_dir)
    nlf_values = load_sidd_nlf(nlf_csv) if nlf_csv is not None else {}
    scene_dirs = sorted(
        path
        for path in source.rglob("*")
        if path.is_dir() and _SCENE_PATTERN.fullmatch(path.name)
    )
    if not scene_dirs:
        raise ValueError(f"{source}: no SIDD scene directories found")
    if held_out_scenes is not None:
        # 只比较完整场景实例名，不能只按三位 scene_id 拒绝：官方训练实例与
        # benchmark 实例会共享物理场景号，但拍摄条件和实例编号不同。
        held_out_names = {
            (
                f"{scene.instance_id}_{scene.scene_id}_{scene.camera_id}_{scene.iso:05d}_"
                f"{scene.shutter_denominator:05d}_{scene.cct:04d}_{scene.brightness}"
            )
            for scene in load_sidd_scene_order(held_out_scenes)
        }
        leaked = sorted(path.name for path in scene_dirs if path.name in held_out_names)
        if leaked:
            raise ValueError(
                "SIDD source contains official held-out benchmark scenes; "
                f"remove them before import: {leaked}"
            )

    sample_dir = output / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
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
            source_pair_id = f"sidd_{scene.instance_id}_{pair_index:03d}"
            if patch_size is None:
                patch_specs: list[tuple[int | None, int, int]] = [(None, 0, 0)]
            else:
                coordinates = _patch_coordinates(
                    height=int(noisy.shape[-2]),
                    width=int(noisy.shape[-1]),
                    patch_size=patch_size,
                    patch_count=patches_per_pair,
                    seed_token=(
                        f"{patch_seed}:{scene_dir.name}:{noisy_path.name}:"
                        f"{target_path.name}"
                    ),
                )
                patch_specs = [
                    (patch_index, top, left)
                    for patch_index, (top, left) in enumerate(coordinates, start=1)
                ]
            source_input_sha256 = _sha256(noisy_path)
            source_target_sha256 = _sha256(target_path)
            for patch_index, top, left in patch_specs:
                if patch_index is None:
                    sample_id = source_pair_id
                    sample_input = noisy
                    sample_target = target
                else:
                    sample_id = f"{source_pair_id}_p{patch_index:03d}"
                    sample_input = noisy[:, top : top + patch_size, left : left + patch_size]
                    sample_target = target[:, top : top + patch_size, left : left + patch_size]
                converted_input = sample_dir / f"{sample_id}_input.npz"
                converted_target = sample_dir / f"{sample_id}_target.npz"
                _save_raw_npz(converted_input, sample_input)
                _save_raw_npz(converted_target, sample_target)
                metadata: dict[str, object] = {
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
                    "source_pair_id": source_pair_id,
                    "source_input_sha256": source_input_sha256,
                    "source_target_sha256": source_target_sha256,
                }
                if patch_index is not None:
                    metadata.update(
                        {
                            "patch_index": patch_index,
                            "patch_top": top,
                            "patch_left": left,
                            "patch_size": patch_size,
                            "patch_seed": patch_seed,
                        }
                    )
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
                        metadata=metadata,
                    )
                )
    manifest_path = output / "manifest.jsonl"
    write_manifest(records, manifest_path)
    return manifest_path


def import_sidd_validation_blocks(
    noisy_mat: str | Path,
    ground_truth_mat: str | Path,
    output_dir: str | Path,
    *,
    scene_order: str | Path,
    nlf_csv: str | Path | None = None,
    split: str = "test",
) -> Path:
    """把官方 SIDD RAW 验证块转换为带 CFA/场景来源的项目清单。

    为控制峰值内存，先加载并写出全部 noisy block，释放数组后再处理 GT。
    输入/GT 的 MAT 变量形状在加载前必须完全一致；第一维必须匹配版本化的
    40 场景顺序，第二维通常为官方规定的 32 个 256×256 Bayer block。
    """

    if split not in {"val", "test", "golden"}:
        raise ValueError("SIDD validation split must be val, test, or golden")
    noisy_path = Path(noisy_mat)
    target_path = Path(ground_truth_mat)
    scene_order_path = Path(scene_order)
    scenes = load_sidd_scene_order(scene_order_path)
    noisy_shape = _sidd_mat_shape(noisy_path, _VALIDATION_NOISY_VARIABLE)
    target_shape = _sidd_mat_shape(target_path, _VALIDATION_GT_VARIABLE)
    if noisy_shape != target_shape:
        raise ValueError(
            f"SIDD validation noisy/GT shape mismatch: {noisy_shape} != {target_shape}"
        )
    if len(noisy_shape) != 4 or noisy_shape[0] != len(scenes):
        raise ValueError(
            f"SIDD validation shape {noisy_shape} does not match {len(scenes)} scenes"
        )

    output = Path(output_dir)
    sample_dir = output / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    noisy_blocks = _load_sidd_block_array(noisy_path, _VALIDATION_NOISY_VARIABLE)
    _write_sidd_block_role(noisy_blocks, scenes, sample_dir, "input")
    del noisy_blocks
    target_blocks = _load_sidd_block_array(target_path, _VALIDATION_GT_VARIABLE)
    _write_sidd_block_role(target_blocks, scenes, sample_dir, "target")
    del target_blocks

    nlf_values = load_sidd_nlf(nlf_csv) if nlf_csv is not None else {}
    noisy_sha256 = _sha256(noisy_path)
    target_sha256 = _sha256(target_path)
    order_sha256 = _sha256(scene_order_path)
    block_count = noisy_shape[1]
    records: list[ManifestRecord] = []
    for scene in scenes:
        scene_name = (
            f"{scene.instance_id}_{scene.scene_id}_{scene.camera_id}_{scene.iso:05d}_"
            f"{scene.shutter_denominator:05d}_{scene.cct:04d}_{scene.brightness}"
        )
        nlf = nlf_values.get(scene_name)
        for block_index in range(1, block_count + 1):
            sample_id = f"sidd_val_{scene.instance_id}_{block_index:02d}"
            records.append(
                ManifestRecord(
                    sample_id=sample_id,
                    dataset_id="sidd",
                    input_path=(Path("samples") / f"{sample_id}_input.npz").as_posix(),
                    target_path=(Path("samples") / f"{sample_id}_target.npz").as_posix(),
                    split=split,
                    sensor_id=f"sidd_{scene.camera_id}",
                    mode="single",
                    session_id="sidd_validation",
                    scene_id=scene.scene_id,
                    iso_bucket=_iso_bucket(scene.iso),
                    metadata={
                        "noise_sigma": _noise_sigma(nlf),
                        "exposure_ratio": 1.0,
                        "wb_rg": 1.0,
                        "wb_bg": 1.0,
                        "cfa_pattern": SIDD_CFA_PATTERNS[scene.camera_id],
                        "iso": scene.iso,
                        "shutter_denominator": scene.shutter_denominator,
                        "cct": scene.cct,
                        "illuminant_brightness": scene.brightness,
                        "validation_block_index": block_index,
                        "nlf": list(nlf) if nlf is not None else None,
                        "source_input_sha256": noisy_sha256,
                        "source_target_sha256": target_sha256,
                        "source_scene_order_sha256": order_sha256,
                    },
                )
            )
    manifest_path = output / "manifest.jsonl"
    write_manifest(records, manifest_path)
    return manifest_path
