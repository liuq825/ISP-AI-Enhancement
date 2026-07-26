"""通过 HTTP Range 从超大 SIDD 场景 ZIP 中只提取一个训练帧。

官方 SIDD Full 每个场景的 noisy/GT ZIP 可达数 GB，完整下载后只取一帧会浪费大量
带宽。``remotezip`` 先读取 ZIP 中央目录，再按成员压缩区间发起 Range GET。本模块在
其上增加场景白名单、held-out 隔离、CRC/大小复核、SHA256 与原子落盘。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zlib
from collections.abc import Callable
from pathlib import Path
from typing import Any
from zipfile import ZipFile, ZipInfo

from isp_ai_enhancement.config import load_yaml

from .sidd import SIDDScene, load_sidd_scene_order

ArchiveFactory = Callable[[str], ZipFile]


def _sha256_and_crc32(path: Path) -> tuple[str, int]:
    """单次分块读取同时计算 SHA256 与 ZIP 使用的 CRC32。"""

    digest = hashlib.sha256()
    crc32 = 0
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
            crc32 = zlib.crc32(chunk, crc32)
    return digest.hexdigest(), crc32 & 0xFFFFFFFF


def _default_archive_factory(url: str) -> ZipFile:
    """延迟导入可选依赖，并以适合中央目录的缓冲区打开远程 ZIP。"""

    try:
        from remotezip import RemoteZip
    except ImportError as error:
        raise RuntimeError(
            "远程 SIDD ZIP 提取需要 acquisition 依赖："
            "pip install -e '.[acquisition]'"
        ) from error
    # support_suffix_range=True 避免先对 Codalab 的预签名 GET URL 发 HEAD；
    # 该镜像的 HEAD 会因方法不同触发签名错误，但 Range GET 已实际验证可用。
    return RemoteZip(
        url,
        timeout=120,
        initial_buffer_size=256 * 1024,
        support_suffix_range=True,
    )


def _find_member(archive: ZipFile, basename: str) -> ZipInfo:
    """按不区分大小写的 basename 找唯一成员，拒绝缺失或歧义 ZIP。"""

    expected = basename.casefold()
    matches = [
        info
        for info in archive.infolist()
        if not info.is_dir() and Path(info.filename).name.casefold() == expected
    ]
    if len(matches) != 1:
        names = [info.filename for info in matches]
        raise ValueError(f"远程 ZIP 中应唯一包含 {basename!r}，实际匹配 {names}")
    return matches[0]


def _extract_member(
    archive: ZipFile,
    info: ZipInfo,
    destination: Path,
    *,
    max_member_bytes: int,
) -> dict[str, Any]:
    """校验并原子提取一个成员；已存在且 CRC 正确时安全复用。"""

    if info.file_size <= 0 or info.file_size > max_member_bytes:
        raise ValueError(
            f"{info.filename}: 解压大小 {info.file_size} 超出 (0,{max_member_bytes}]"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        sha256, crc32 = _sha256_and_crc32(destination)
        if destination.stat().st_size != info.file_size or crc32 != info.CRC:
            raise ValueError(f"已有文件与远程 ZIP 大小/CRC 不一致：{destination}")
        return {
            "member": info.filename,
            "member_bytes": info.file_size,
            "compressed_bytes": info.compress_size,
            "crc32": f"{info.CRC:08x}",
            "sha256": sha256,
            "reused_existing": True,
        }

    temporary = destination.with_name(f"{destination.name}.partial")
    temporary.unlink(missing_ok=True)
    try:
        # ZipExtFile 在完整读取后会校验中央目录 CRC；随后再计算本地 CRC/SHA256，
        # 避免网络、解压或磁盘中断被误标为可用训练文件。
        with archive.open(info, "r") as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target, length=4 * 1024 * 1024)
        sha256, crc32 = _sha256_and_crc32(temporary)
        if temporary.stat().st_size != info.file_size or crc32 != info.CRC:
            raise ValueError(f"{info.filename}: 提取后大小或 CRC 校验失败")
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "member": info.filename,
        "member_bytes": info.file_size,
        "compressed_bytes": info.compress_size,
        "crc32": f"{info.CRC:08x}",
        "sha256": sha256,
        "reused_existing": False,
    }


def _scene_name(scene: SIDDScene) -> str:
    """把已解析的场景元数据还原为官方规范目录名。"""

    return (
        f"{scene.instance_id}_{scene.scene_id}_{scene.camera_id}_{scene.iso:05d}_"
        f"{scene.shutter_denominator:05d}_{scene.cct:04d}_{scene.brightness}"
    )


def fetch_sidd_raw_pair(
    *,
    scene_name: str,
    noisy_zip_url: str,
    ground_truth_zip_url: str,
    frame_index: int,
    output_dir: str | Path,
    held_out_scenes: str | Path,
    max_member_bytes: int = 512 * 1024 * 1024,
    archive_factory: ArchiveFactory | None = None,
) -> Path:
    """从两个远程场景 ZIP 提取同编号 RAW 配对并写审计收据。

    ``archive_factory`` 只用于测试或受控镜像适配；生产默认使用 RemoteZip。
    每个角色先按 ZIP CRC 与本地 SHA256 验证，再原子替换最终 MAT。若上次仅完成
    noisy，重试会复核并复用它，只继续获取 GT。
    """

    if frame_index <= 0 or frame_index > 999:
        raise ValueError("frame_index 必须在 [1,999]")
    scene = SIDDScene.from_directory(Path(scene_name))
    canonical_name = _scene_name(scene)
    held_out_names = {_scene_name(value) for value in load_sidd_scene_order(held_out_scenes)}
    if canonical_name in held_out_names:
        raise ValueError(f"拒绝获取官方 held-out benchmark 场景：{canonical_name}")
    if not noisy_zip_url.startswith(("http://", "https://")):
        raise ValueError("noisy_zip_url 必须是 HTTP(S) URL")
    if not ground_truth_zip_url.startswith(("http://", "https://")):
        raise ValueError("ground_truth_zip_url 必须是 HTTP(S) URL")

    factory = archive_factory or _default_archive_factory
    output = Path(output_dir)
    scene_dir = output / canonical_name
    instance = scene.instance_id
    frame = f"{frame_index:03d}"
    noisy_basename = f"{instance}_NOISY_RAW_{frame}.MAT"
    target_basename = f"{instance}_GT_RAW_{frame}.MAT"

    with factory(noisy_zip_url) as archive:
        noisy_info = _find_member(archive, noisy_basename)
        noisy_receipt = _extract_member(
            archive,
            noisy_info,
            scene_dir / noisy_basename,
            max_member_bytes=max_member_bytes,
        )
    with factory(ground_truth_zip_url) as archive:
        target_info = _find_member(archive, target_basename)
        target_receipt = _extract_member(
            archive,
            target_info,
            scene_dir / target_basename,
            max_member_bytes=max_member_bytes,
        )

    receipt = {
        "format_version": 1,
        "scene_name": canonical_name,
        "frame_index": frame_index,
        "noisy_zip_url": noisy_zip_url,
        "ground_truth_zip_url": ground_truth_zip_url,
        "noisy": noisy_receipt,
        "ground_truth": target_receipt,
    }
    receipt_path = scene_dir / f"{instance}_RAW_{frame}.receipt.json"
    temporary_receipt = receipt_path.with_name(f"{receipt_path.name}.tmp")
    temporary_receipt.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_receipt.replace(receipt_path)
    return receipt_path


def _validate_subset_spec(config_path: Path) -> tuple[int, list[dict[str, str]]]:
    """一次性校验子集配置，确保发起首个网络请求前已发现全部格式问题。"""

    values = load_yaml(config_path)
    frame_index = values.get("frame_index")
    raw_scenes = values.get("scenes")
    if not isinstance(frame_index, int) or not 1 <= frame_index <= 999:
        raise ValueError(f"{config_path}: frame_index 必须是 [1,999] 内的整数")
    if not isinstance(raw_scenes, list) or not raw_scenes:
        raise ValueError(f"{config_path}: scenes 必须是非空列表")

    scenes: list[dict[str, str]] = []
    seen_names: set[str] = set()
    for index, raw_scene in enumerate(raw_scenes):
        if not isinstance(raw_scene, dict):
            raise ValueError(f"{config_path}: scenes[{index}] 必须是映射")
        required = ("scene", "noisy_url", "ground_truth_url")
        if any(not isinstance(raw_scene.get(key), str) for key in required):
            raise ValueError(f"{config_path}: scenes[{index}] 缺少字符串字段 {required}")
        scene = {key: str(raw_scene[key]) for key in required}
        # 解析规范场景名并预检 URL，避免前几个场景下载完成后才发现末尾配置错误。
        canonical_name = _scene_name(SIDDScene.from_directory(Path(scene["scene"])))
        scene["scene"] = canonical_name
        if canonical_name in seen_names:
            raise ValueError(f"{config_path}: 重复场景 {canonical_name}")
        if not scene["noisy_url"].startswith(("http://", "https://")):
            raise ValueError(f"{config_path}: {scene['scene']} noisy_url 不是 HTTP(S)")
        if not scene["ground_truth_url"].startswith(("http://", "https://")):
            raise ValueError(f"{config_path}: {scene['scene']} ground_truth_url 不是 HTTP(S)")
        seen_names.add(canonical_name)
        scenes.append(scene)
    return frame_index, scenes


def fetch_sidd_raw_subset(
    *,
    config: str | Path,
    output_dir: str | Path,
    held_out_scenes: str | Path,
    max_member_bytes: int = 512 * 1024 * 1024,
    archive_factory: ArchiveFactory | None = None,
) -> Path:
    """按版本化配置顺序获取多场景 RAW 配对，并生成集合级可审计收据。

    单场景函数负责 CRC、SHA256、原子写入和断点复用；本函数先完整校验配置，
    再顺序执行，避免并发 Range 请求压垮公共镜像。任一场景失败时不会伪造集合
    成功收据，已完成场景可在下次运行中经 CRC 复核后复用。
    """

    config_path = Path(config)
    frame_index, scenes = _validate_subset_spec(config_path)
    config_sha256, _config_crc32 = _sha256_and_crc32(config_path)
    held_out_names = {
        _scene_name(scene) for scene in load_sidd_scene_order(held_out_scenes)
    }
    leaked = sorted(scene["scene"] for scene in scenes if scene["scene"] in held_out_names)
    if leaked:
        # 批量入口在首个网络请求前检查全部行；不能让配置末尾的泄漏场景导致
        # 已下载一半后才失败，也不能依赖调用者记住验证集名单。
        raise ValueError(f"子集配置包含官方 held-out benchmark 场景：{leaked}")
    output = Path(output_dir)
    receipts: list[dict[str, Any]] = []
    for scene in scenes:
        receipt_path = fetch_sidd_raw_pair(
            scene_name=scene["scene"],
            noisy_zip_url=scene["noisy_url"],
            ground_truth_zip_url=scene["ground_truth_url"],
            frame_index=frame_index,
            output_dir=output,
            held_out_scenes=held_out_scenes,
            max_member_bytes=max_member_bytes,
            archive_factory=archive_factory,
        )
        receipts.append(
            {
                "scene_name": scene["scene"],
                "receipt": receipt_path.relative_to(output).as_posix(),
                "receipt_sha256": _sha256_and_crc32(receipt_path)[0],
            }
        )

    collection_receipt = {
        "format_version": 1,
        "config": str(config_path),
        "config_sha256": config_sha256,
        "frame_index": frame_index,
        "scene_count": len(receipts),
        "scenes": receipts,
    }
    receipt_path = output / "subset.receipt.json"
    output.mkdir(parents=True, exist_ok=True)
    temporary = receipt_path.with_name(f"{receipt_path.name}.tmp")
    temporary.write_text(
        json.dumps(collection_receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(receipt_path)
    return receipt_path
