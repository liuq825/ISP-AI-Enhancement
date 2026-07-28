"""官方 SIDD 场景级 RAW 数据发现、校验与项目格式转换。

SIDD 的 ``*_NOISY_RAW``/``*_GT_RAW`` 文件是二维 Bayer 马赛克，而项目网络
接收统一顺序的四通道 packed RAW。本模块根据官方相机 CFA 表完成打包，
保留源文件 SHA256、NLF、ISO 和色温等来源信息，并按场景划分数据集。
"""

from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

import numpy as np
import torch
import yaml

from isp_ai_enhancement.config import load_yaml

from .context import canonical_pack_bayer
from .governance import validate_data_requirements
from .manifest import ManifestRecord, read_manifest, validate_manifest, write_manifest

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
    """原子写入 packed RAW NPZ；完整旧文件逐元素一致时直接复用。

    Medium 规模 patch 导入会产生上万个文件。任务中断后重跑时，已完成 NPZ 必须
    解压并与当前确定性结果逐元素核对；一致则保留原文件，不一致则硬失败，避免
    静默覆盖可能来自不同 seed、CFA 或源文件的产物。
    """

    expected = np.asarray(raw, dtype=np.float32)
    if path.is_file():
        try:
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != {"raw"}:
                    raise ValueError("NPZ 字段必须且只能包含 raw")
                existing = np.asarray(archive["raw"])
        except (OSError, ValueError) as error:
            raise ValueError(f"{path}: 已有 NPZ 无法安全复用：{error}") from error
        if existing.dtype != np.float32 or not np.array_equal(existing, expected):
            raise ValueError(f"{path}: 已有 NPZ 与当前确定性转换结果不一致")
        return
    temporary = path.with_name(f"{path.name}.tmp.npz")
    np.savez_compressed(temporary, raw=expected)
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


def _load_audited_raw(path: Path) -> np.ndarray:
    """读取一个导入后的 NPZ，并执行训练输入所需的严格数值约束。

    审计不接受额外字段，因为对象数组或旁路元数据会扩大反序列化攻击面，也可能
    让训练与审计读取到不同内容。这里不做裁剪或容错修正；任何异常都表示导入产物
    已损坏、被改写或不再符合 ``4×H×W float32`` RAW 契约。
    """

    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"raw"}:
                raise ValueError("NPZ 字段必须且只能包含 raw")
            raw = np.asarray(archive["raw"])
    except (BadZipFile, OSError, ValueError) as error:
        raise ValueError(f"{path}: 无法审计导入后的 RAW：{error}") from error
    if raw.dtype != np.float32:
        raise ValueError(f"{path}: RAW dtype 必须为 float32，实际为 {raw.dtype}")
    if raw.ndim != 3 or raw.shape[0] != 4 or min(raw.shape[1:]) <= 0:
        raise ValueError(f"{path}: RAW 必须为 4×H×W，实际为 {raw.shape}")
    if not np.isfinite(raw).all():
        raise ValueError(f"{path}: RAW 含 NaN 或无穷值")
    if float(raw.min()) < -1e-6 or float(raw.max()) > 1.0 + 1e-6:
        raise ValueError(f"{path}: RAW 数值必须位于 [0, 1]")
    return raw


def _portable_audit_path(path: Path) -> str:
    """优先把当前工作区内的路径写成 POSIX 相对路径，避免回执绑定 Windows 盘符。"""

    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def write_sidd_import_audit_receipt(
    manifest_path: str | Path,
    training_config: str | Path,
    destination: str | Path,
    *,
    acquisition_receipt: str | Path | None = None,
    nlf_csv: str | Path | None = None,
) -> Path:
    """全量复核 packed RAW 导入产物并写出可提交 Git 的确定性摘要回执。

    文件级 SHA256 会受 NPZ 压缩器和 ZIP 时间戳影响，因此最终
    ``array_content_sha256`` 按“相对路径、dtype、shape、解压数组字节”顺序累计。
    只要训练真正读取的数值未变化，即使重新压缩或跨平台复制，内容摘要仍保持一致。
    同时复核 Manifest 防泄漏规则、源配对身份、patch 序号以及正式训练配置声明的
    数据充分性门槛，防止仅凭文件数量误判数据已经可用于 P0。
    """

    manifest = Path(manifest_path)
    config_path = Path(training_config)
    target = Path(destination)
    records = read_manifest(manifest)
    manifest_errors = validate_manifest(records, root=manifest.parent)
    if manifest_errors:
        formatted = "\n".join(f"- {error}" for error in manifest_errors)
        raise ValueError(f"SIDD 导入 Manifest 审计失败：\n{formatted}")

    config = load_yaml(config_path)
    if "data_requirements" not in config:
        raise ValueError(f"{config_path}: 正式导入审计必须声明 data_requirements")
    requirement_errors = validate_data_requirements(
        records, config.get("data_requirements")
    )
    if requirement_errors:
        formatted = "\n".join(f"- {error}" for error in requirement_errors)
        raise ValueError(f"SIDD 导入数据充分性审计失败：\n{formatted}")

    root = manifest.parent.resolve()
    referenced_paths: set[str] = set()
    split_records: Counter[str] = Counter()
    split_source_pairs: dict[str, set[str]] = defaultdict(set)
    split_scene_groups: dict[str, set[tuple[str, str]]] = defaultdict(set)
    split_sensors: dict[str, set[str]] = defaultdict(set)
    split_iso_buckets: dict[str, set[str]] = defaultdict(set)
    split_modes: dict[str, set[str]] = defaultdict(set)
    source_pair_patches: dict[str, set[int]] = defaultdict(set)
    source_pair_hashes: dict[str, tuple[str, str]] = {}
    patch_sizes: set[int] = set()
    patch_seeds: set[int] = set()
    shape_counts: Counter[str] = Counter()
    compressed_bytes = 0
    array_bytes = 0
    content_digest = hashlib.sha256()

    for record in sorted(records, key=lambda item: item.sample_id):
        if record.dataset_id != "sidd":
            raise ValueError(
                f"{record.sample_id}: audit-sidd-import 只接受 dataset_id=sidd"
            )
        source_pair_id = str(record.metadata.get("source_pair_id", "")).strip()
        if not source_pair_id:
            raise ValueError(f"{record.sample_id}: 缺少 metadata.source_pair_id")
        source_input_sha256 = str(
            record.metadata.get("source_input_sha256", "")
        ).lower()
        source_target_sha256 = str(
            record.metadata.get("source_target_sha256", "")
        ).lower()
        if not re.fullmatch(r"[0-9a-f]{64}", source_input_sha256) or not re.fullmatch(
            r"[0-9a-f]{64}", source_target_sha256
        ):
            raise ValueError(f"{record.sample_id}: 源 RAW SHA256 非法或缺失")
        source_hashes = (source_input_sha256, source_target_sha256)
        previous_hashes = source_pair_hashes.setdefault(source_pair_id, source_hashes)
        if previous_hashes != source_hashes:
            raise ValueError(f"{source_pair_id}: 派生 patch 的源 RAW SHA256 不一致")

        try:
            patch_index = int(record.metadata["patch_index"])
            patch_size = int(record.metadata["patch_size"])
            patch_seed = int(record.metadata["patch_seed"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{record.sample_id}: patch 元数据非法或缺失") from error
        if patch_index <= 0 or patch_size <= 0:
            raise ValueError(f"{record.sample_id}: patch_index/patch_size 必须为正整数")
        if patch_index in source_pair_patches[source_pair_id]:
            raise ValueError(f"{source_pair_id}: patch_index={patch_index} 重复")
        source_pair_patches[source_pair_id].add(patch_index)
        patch_sizes.add(patch_size)
        patch_seeds.add(patch_seed)

        split_records[record.split] += 1
        split_source_pairs[record.split].add(source_pair_id)
        split_scene_groups[record.split].add((record.session_id, record.scene_id))
        split_sensors[record.split].add(record.sensor_id)
        split_iso_buckets[record.split].add(record.iso_bucket)
        split_modes[record.split].add(record.mode)

        pair_shapes: list[tuple[int, ...]] = []
        for role, raw_path in (
            ("input", record.input_path),
            ("target", record.target_path),
        ):
            candidate = Path(raw_path)
            candidate = candidate if candidate.is_absolute() else manifest.parent / candidate
            resolved = candidate.resolve()
            try:
                relative = resolved.relative_to(root).as_posix()
            except ValueError as error:
                raise ValueError(
                    f"{record.sample_id}: {role} 文件必须位于 Manifest 目录内"
                ) from error
            if relative in referenced_paths:
                raise ValueError(f"{record.sample_id}: 重复引用文件 {relative}")
            referenced_paths.add(relative)

            raw = _load_audited_raw(resolved)
            shape = tuple(int(value) for value in raw.shape)
            if shape[-2:] != (patch_size, patch_size):
                raise ValueError(
                    f"{record.sample_id}: {role} shape={shape} "
                    f"与 patch_size={patch_size} 不一致"
                )
            pair_shapes.append(shape)
            shape_counts["x".join(str(value) for value in shape)] += 1
            compressed_bytes += resolved.stat().st_size
            array_bytes += raw.nbytes

            # 摘要显式加入角色路径、类型和形状，避免简单拼接数组字节产生边界歧义。
            content_digest.update(relative.encode("utf-8"))
            content_digest.update(b"\0")
            content_digest.update(str(raw.dtype).encode("ascii"))
            content_digest.update(b"\0")
            content_digest.update(",".join(str(value) for value in shape).encode("ascii"))
            content_digest.update(b"\0")
            content_digest.update(raw.tobytes(order="C"))
        if pair_shapes[0] != pair_shapes[1]:
            raise ValueError(
                f"{record.sample_id}: input/target shape 不一致 "
                f"{pair_shapes[0]} != {pair_shapes[1]}"
            )

    patch_counts = {len(indices) for indices in source_pair_patches.values()}
    for source_pair_id, indices in source_pair_patches.items():
        expected = set(range(1, len(indices) + 1))
        if indices != expected:
            raise ValueError(f"{source_pair_id}: patch_index 必须从 1 连续编号")

    source_digest = hashlib.sha256()
    for source_pair_id, (input_sha256, target_sha256) in sorted(
        source_pair_hashes.items()
    ):
        source_digest.update(
            f"{source_pair_id}\t{input_sha256}\t{target_sha256}\n".encode()
        )

    def counts(values: dict[str, set[Any]]) -> dict[str, int]:
        """把 split→集合转换为按 split 排序的稳定计数映射。"""

        return {split: len(items) for split, items in sorted(values.items())}

    source_evidence: dict[str, object] = {}
    for field_name, value in (
        ("acquisition_receipt", acquisition_receipt),
        ("nlf_csv", nlf_csv),
    ):
        if value is None:
            continue
        evidence_path = Path(value)
        if not evidence_path.is_file():
            raise ValueError(f"{evidence_path}: 来源证据文件不存在")
        source_evidence[field_name] = _portable_audit_path(evidence_path)
        source_evidence[f"{field_name}_sha256"] = _sha256(evidence_path)

    receipt = {
        "receipt_version": 1,
        "checked_on": date.today().isoformat(),
        "dataset_id": "sidd",
        "source_evidence": source_evidence,
        "import": {
            "manifest": _portable_audit_path(manifest),
            "manifest_sha256": _sha256(manifest),
            "training_config": _portable_audit_path(config_path),
            "training_config_sha256": _sha256(config_path),
            "patch_sizes": sorted(patch_sizes),
            "patch_seeds": sorted(patch_seeds),
            "patches_per_source_pair": sorted(patch_counts),
        },
        "dataset": {
            "record_count": len(records),
            "referenced_npz_count": len(referenced_paths),
            "compressed_npz_bytes": compressed_bytes,
            "uncompressed_array_bytes": array_bytes,
            "array_shapes": dict(sorted(shape_counts.items())),
            "array_content_sha256": content_digest.hexdigest(),
            "source_pair_count": len(source_pair_hashes),
            "source_pair_identity_sha256": source_digest.hexdigest(),
            "split_records": dict(sorted(split_records.items())),
            "split_source_pairs": counts(split_source_pairs),
            "split_scene_groups": counts(split_scene_groups),
            "train_sensors": sorted(split_sensors["train"]),
            "train_iso_buckets": sorted(split_iso_buckets["train"]),
            "train_modes": sorted(split_modes["train"]),
        },
        "verification": {
            "manifest_passed": True,
            "data_requirements_passed": True,
            "arrays_single_raw_float32_finite_normalized": True,
            "input_target_shapes_match": True,
        },
        "limitations": [
            "回执证明公开 SIDD packed RAW 导入完整性，不代表模型已经训练收敛",
            "公开数据充分性门槛通过不替代目标 Sensor、目标 DDK 与麒麟 9000 真机证据",
        ],
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.tmp")
    temporary.write_text(
        "# SIDD Medium packed RAW 导入审计回执；NPZ 数据本体不进入 Git。\n"
        + yaml.safe_dump(receipt, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(target)
    return target


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
