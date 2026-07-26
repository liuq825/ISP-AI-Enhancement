"""通过 HTTP Range 从超大 SIDD 场景 ZIP 中只提取一个训练帧。

官方 SIDD Full 每个场景的 noisy/GT ZIP 可达数 GB，完整下载后只取一帧会浪费大量
带宽。``remotezip`` 先读取 ZIP 中央目录，再按成员压缩区间发起 Range GET。本模块在
其上增加场景白名单、held-out 隔离、CRC/大小复核、SHA256 与原子落盘。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
import zlib
from collections.abc import Callable
from pathlib import Path
from typing import Any
from zipfile import ZipFile, ZipInfo

from isp_ai_enhancement.config import load_yaml

from .sidd import SIDDScene, load_sidd_scene_order

# 两个可注入接口分别隔离网络归档创建和人类可读进度，单测无需访问公网。
ArchiveFactory = Callable[[str], ZipFile]
ProgressCallback = Callable[[str], None]


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


def fetch_sidd_raw_frames(
    *,
    scene_name: str,
    noisy_zip_url: str,
    ground_truth_zip_url: str,
    frame_indices: list[int] | tuple[int, ...],
    output_dir: str | Path,
    held_out_scenes: str | Path,
    max_member_bytes: int = 512 * 1024 * 1024,
    archive_factory: ArchiveFactory | None = None,
) -> list[Path]:
    """一次打开两个远程 ZIP，提取同场景的多组 RAW 配对并写审计收据。

    ``archive_factory`` 只用于测试或受控镜像适配；生产默认使用 RemoteZip。
    每个角色先按 ZIP CRC 与本地 SHA256 验证，再原子替换最终 MAT。多帧共享一次
    中央目录读取，避免 320 对配置重复打开同一场景归档。若上次只完成部分文件，
    重试会逐一复核并复用，只继续缺失成员。
    """

    frames = list(frame_indices)
    if (
        not frames
        or any(
            not isinstance(frame, int)
            or isinstance(frame, bool)
            or not 1 <= frame <= 999
            for frame in frames
        )
        or len(frames) != len(set(frames))
    ):
        raise ValueError("frame_indices 必须是 [1,999] 内的不重复整数序列")
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

    noisy_receipts: dict[int, dict[str, Any]] = {}
    with factory(noisy_zip_url) as archive:
        for frame_index in frames:
            frame = f"{frame_index:03d}"
            basename = f"{instance}_NOISY_RAW_{frame}.MAT"
            noisy_receipts[frame_index] = _extract_member(
                archive,
                _find_member(archive, basename),
                scene_dir / basename,
                max_member_bytes=max_member_bytes,
            )
    target_receipts: dict[int, dict[str, Any]] = {}
    with factory(ground_truth_zip_url) as archive:
        for frame_index in frames:
            frame = f"{frame_index:03d}"
            basename = f"{instance}_GT_RAW_{frame}.MAT"
            target_receipts[frame_index] = _extract_member(
                archive,
                _find_member(archive, basename),
                scene_dir / basename,
                max_member_bytes=max_member_bytes,
            )

    receipt_paths: list[Path] = []
    for frame_index in frames:
        frame = f"{frame_index:03d}"
        receipt = {
            "format_version": 1,
            "scene_name": canonical_name,
            "frame_index": frame_index,
            "noisy_zip_url": noisy_zip_url,
            "ground_truth_zip_url": ground_truth_zip_url,
            "noisy": noisy_receipts[frame_index],
            "ground_truth": target_receipts[frame_index],
        }
        receipt_path = scene_dir / f"{instance}_RAW_{frame}.receipt.json"
        temporary_receipt = receipt_path.with_name(f"{receipt_path.name}.tmp")
        temporary_receipt.write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_receipt.replace(receipt_path)
        receipt_paths.append(receipt_path)
    return receipt_paths


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
    """提取单个 RAW 帧配对；实现复用多帧入口的全部安全校验。"""

    return fetch_sidd_raw_frames(
        scene_name=scene_name,
        noisy_zip_url=noisy_zip_url,
        ground_truth_zip_url=ground_truth_zip_url,
        frame_indices=(frame_index,),
        output_dir=output_dir,
        held_out_scenes=held_out_scenes,
        max_member_bytes=max_member_bytes,
        archive_factory=archive_factory,
    )[0]


def _validate_subset_spec(config_path: Path) -> tuple[list[int], list[dict[str, str]]]:
    """一次性校验子集配置，确保发起首个网络请求前已发现全部格式问题。"""

    values = load_yaml(config_path)
    has_single_frame = "frame_index" in values
    has_multiple_frames = "frame_indices" in values
    if has_single_frame == has_multiple_frames:
        raise ValueError(f"{config_path}: 必须且只能定义 frame_index 或 frame_indices")
    raw_frames = (
        [values["frame_index"]]
        if has_single_frame
        else values["frame_indices"]
    )
    if (
        not isinstance(raw_frames, list)
        or not raw_frames
        or any(
            not isinstance(frame, int)
            or isinstance(frame, bool)
            or not 1 <= frame <= 999
            for frame in raw_frames
        )
    ):
        raise ValueError(f"{config_path}: 帧编号必须是 [1,999] 内的非空整数列表")
    frame_indices = [int(frame) for frame in raw_frames]
    if len(frame_indices) != len(set(frame_indices)):
        raise ValueError(f"{config_path}: frame_indices 含重复帧")
    raw_scenes = values.get("scenes")
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
        fallback_fields = ("fallback_noisy_url", "fallback_ground_truth_url")
        has_fallback = [key in raw_scene for key in fallback_fields]
        if any(has_fallback) and not all(has_fallback):
            raise ValueError(
                f"{config_path}: scenes[{index}] 必须同时定义 {fallback_fields}"
            )
        if all(has_fallback):
            if any(not isinstance(raw_scene.get(key), str) for key in fallback_fields):
                raise ValueError(
                    f"{config_path}: scenes[{index}] 的备用 URL 必须是字符串"
                )
            scene.update({key: str(raw_scene[key]) for key in fallback_fields})
        # 解析规范场景名并预检 URL，避免前几个场景下载完成后才发现末尾配置错误。
        canonical_name = _scene_name(SIDDScene.from_directory(Path(scene["scene"])))
        scene["scene"] = canonical_name
        if canonical_name in seen_names:
            raise ValueError(f"{config_path}: 重复场景 {canonical_name}")
        if not scene["noisy_url"].startswith(("http://", "https://")):
            raise ValueError(f"{config_path}: {scene['scene']} noisy_url 不是 HTTP(S)")
        if not scene["ground_truth_url"].startswith(("http://", "https://")):
            raise ValueError(f"{config_path}: {scene['scene']} ground_truth_url 不是 HTTP(S)")
        if all(has_fallback):
            if not scene["fallback_noisy_url"].startswith(("http://", "https://")):
                raise ValueError(
                    f"{config_path}: {scene['scene']} fallback_noisy_url 不是 HTTP(S)"
                )
            if not scene["fallback_ground_truth_url"].startswith(
                ("http://", "https://")
            ):
                raise ValueError(
                    f"{config_path}: {scene['scene']} "
                    "fallback_ground_truth_url 不是 HTTP(S)"
                )
            if (
                scene["fallback_noisy_url"] == scene["noisy_url"]
                and scene["fallback_ground_truth_url"] == scene["ground_truth_url"]
            ):
                raise ValueError(f"{config_path}: {scene['scene']} 备用镜像与主镜像相同")
        seen_names.add(canonical_name)
        scenes.append(scene)
    return frame_indices, scenes


def _scene_sources(scene: dict[str, str]) -> list[tuple[str, str]]:
    """按优先级返回同一场景的主镜像和可选备用镜像 URL 对。"""

    sources = [(scene["noisy_url"], scene["ground_truth_url"])]
    if "fallback_noisy_url" in scene:
        sources.append(
            (
                scene["fallback_noisy_url"],
                scene["fallback_ground_truth_url"],
            )
        )
    return sources


def _verified_local_scene_receipts(
    *,
    scene: dict[str, str],
    frame_indices: list[int],
    output: Path,
) -> list[Path] | None:
    """用已有收据离线复核完整场景；收据不齐时返回 ``None`` 继续远程恢复。"""

    instance = scene["scene"].split("_", 1)[0]
    scene_dir = output / scene["scene"]
    receipt_paths = [
        scene_dir / f"{instance}_RAW_{frame_index:03d}.receipt.json"
        for frame_index in frame_indices
    ]
    if not all(path.is_file() for path in receipt_paths):
        return None
    for frame_index, receipt_path in zip(
        frame_indices,
        receipt_paths,
        strict=True,
    ):
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt_source = (
                receipt.get("noisy_zip_url"),
                receipt.get("ground_truth_zip_url"),
            )
            if (
                receipt.get("scene_name") != scene["scene"]
                or receipt.get("frame_index") != frame_index
                or receipt_source not in _scene_sources(scene)
            ):
                raise ValueError("收据身份或来源 URL 与当前配置不一致")
            frame = f"{frame_index:03d}"
            for role, marker in (("noisy", "NOISY"), ("ground_truth", "GT")):
                details = receipt[role]
                mat_path = scene_dir / f"{instance}_{marker}_RAW_{frame}.MAT"
                if (
                    not mat_path.is_file()
                    or mat_path.stat().st_size != int(details["member_bytes"])
                ):
                    raise ValueError(f"{role} MAT 缺失或大小不匹配")
                sha256, crc32 = _sha256_and_crc32(mat_path)
                if sha256 != str(details["sha256"]).casefold():
                    raise ValueError(f"{role} MAT SHA256 不匹配")
                if f"{crc32:08x}" != str(details["crc32"]).casefold():
                    raise ValueError(f"{role} MAT CRC32 不匹配")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"{receipt_path}: 已完成场景离线复核失败：{error}") from error
    return receipt_paths


def fetch_sidd_raw_subset(
    *,
    config: str | Path,
    output_dir: str | Path,
    held_out_scenes: str | Path,
    max_member_bytes: int = 512 * 1024 * 1024,
    archive_factory: ArchiveFactory | None = None,
    progress_callback: ProgressCallback | None = None,
    max_attempts: int = 4,
    retry_backoff_seconds: float = 5.0,
) -> Path:
    """按版本化配置顺序获取多场景 RAW 配对，并生成集合级可审计收据。

    单场景函数负责 CRC、SHA256、原子写入和断点复用；本函数先完整校验配置，
    再顺序执行，避免并发 Range 请求压垮公共镜像。任一场景失败时不会伪造集合
    成功收据，已完成场景可在下次运行中经 CRC 复核后复用。
    """

    config_path = Path(config)
    if (
        not isinstance(max_attempts, int)
        or isinstance(max_attempts, bool)
        or max_attempts <= 0
    ):
        raise ValueError("max_attempts 必须为正整数")
    if retry_backoff_seconds < 0:
        raise ValueError("retry_backoff_seconds 不能为负数")
    frame_indices, scenes = _validate_subset_spec(config_path)
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
    frame_label = ",".join(f"{frame:03d}" for frame in frame_indices)
    for scene_index, scene in enumerate(scenes, start=1):
        label = f"{scene['scene']} frames [{frame_label}]"
        if progress_callback is not None:
            progress_callback(f"[{scene_index}/{len(scenes)}] 开始获取 {label}")
        receipt_paths = _verified_local_scene_receipts(
            scene=scene,
            frame_indices=frame_indices,
            output=output,
        )
        if receipt_paths is not None:
            if progress_callback is not None:
                progress_callback(
                    f"[{scene_index}/{len(scenes)}] 本地 SHA256/CRC 复核通过 {label}"
                )
        else:
            for attempt in range(1, max_attempts + 1):
                source_errors: list[Exception] = []
                sources = _scene_sources(scene)
                for source_index, (noisy_url, ground_truth_url) in enumerate(
                    sources,
                    start=1,
                ):
                    try:
                        receipt_paths = fetch_sidd_raw_frames(
                            scene_name=scene["scene"],
                            noisy_zip_url=noisy_url,
                            ground_truth_zip_url=ground_truth_url,
                            frame_indices=frame_indices,
                            output_dir=output,
                            held_out_scenes=held_out_scenes,
                            max_member_bytes=max_member_bytes,
                            archive_factory=archive_factory,
                        )
                    except ValueError:
                        # 身份、held-out、大小或 CRC 错误是确定性数据问题，切换镜像
                        # 可能掩盖来源错配，必须立即失败。
                        raise
                    except Exception as error:
                        source_errors.append(error)
                        if (
                            progress_callback is not None
                            and source_index < len(sources)
                        ):
                            progress_callback(
                                f"[{scene_index}/{len(scenes)}] 镜像 "
                                f"{source_index}/{len(sources)} "
                                f"{type(error).__name__}: {error}；尝试备用镜像"
                            )
                        continue
                    break
                else:
                    # 同一轮的主/备镜像均失败后才进入场景级退避；下一轮仍从主镜像
                    # 开始，避免永久偏向一次偶发成功但随后损坏的备用端点。
                    error = source_errors[-1]
                    if attempt >= max_attempts:
                        raise error
                    delay = retry_backoff_seconds * attempt
                    if progress_callback is not None:
                        progress_callback(
                            f"[{scene_index}/{len(scenes)}] 全部 {len(sources)} 个镜像"
                            f"均失败，最后错误 {type(error).__name__}: {error}；"
                            f"{delay:g} 秒后进行第 "
                            f"{attempt + 1}/{max_attempts} 次尝试"
                        )
                    # 退避只作用于可恢复的场景级异常，默认最长 15 秒，避免快速压测
                    # 两个公共镜像。
                    if delay > 0:
                        time.sleep(delay)
                    continue
                break
        if receipt_paths is None:  # pragma: no cover - 循环穷尽时异常已在上方重新抛出
            raise RuntimeError("SIDD 场景获取未返回收据")
        for frame_index, receipt_path in zip(
            frame_indices,
            receipt_paths,
            strict=True,
        ):
            receipts.append(
                {
                    "scene_name": scene["scene"],
                    "frame_index": frame_index,
                    "receipt": receipt_path.relative_to(output).as_posix(),
                    "receipt_sha256": _sha256_and_crc32(receipt_path)[0],
                }
            )
        if progress_callback is not None:
            progress_callback(f"[{scene_index}/{len(scenes)}] 已校验 {label}")

    collection_receipt = {
        "format_version": 2,
        "config": str(config_path),
        "config_sha256": config_sha256,
        "frame_indices": frame_indices,
        "pair_count": len(receipts),
        "max_attempts": max_attempts,
        "scene_count": len(scenes),
        "pairs": receipts,
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


def sidd_subset_status(
    *,
    config: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """快速汇总长时间 SIDD 获取进度，不重新计算数百个大文件哈希。

    每个 pair 收据只会在大小、CRC 和 SHA256 已验证后生成，因此状态检查只需确认
    收据身份、记录大小与当前 MAT 大小仍一致。最终发布前仍以获取器重跑或完整收据
    为准；本函数主要供后台进度与异常文件发现使用。
    """

    config_path = Path(config)
    frame_indices, scenes = _validate_subset_spec(config_path)
    output = Path(output_dir)
    completed_pairs = 0
    completed_scene_names: set[str] = set()
    errors: list[str] = []
    for scene in scenes:
        instance = scene["scene"].split("_", 1)[0]
        scene_dir = output / scene["scene"]
        scene_complete = True
        for frame_index in frame_indices:
            frame = f"{frame_index:03d}"
            receipt_path = scene_dir / f"{instance}_RAW_{frame}.receipt.json"
            if not receipt_path.is_file():
                scene_complete = False
                continue
            try:
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                if (
                    receipt.get("scene_name") != scene["scene"]
                    or receipt.get("frame_index") != frame_index
                ):
                    raise ValueError("收据场景或帧编号不匹配")
                for role, marker in (("noisy", "NOISY"), ("ground_truth", "GT")):
                    details = receipt[role]
                    mat_path = scene_dir / f"{instance}_{marker}_RAW_{frame}.MAT"
                    expected_bytes = int(details["member_bytes"])
                    if not mat_path.is_file() or mat_path.stat().st_size != expected_bytes:
                        raise ValueError(f"{role} MAT 缺失或大小不匹配")
                    if len(str(details["sha256"])) != 64:
                        raise ValueError(f"{role} SHA256 字段非法")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                scene_complete = False
                errors.append(f"{receipt_path}: {error}")
                continue
            completed_pairs += 1
        if scene_complete:
            completed_scene_names.add(scene["scene"])

    partials = sorted(
        path.relative_to(output).as_posix()
        for path in output.rglob("*.partial")
        if path.is_file()
    ) if output.is_dir() else []
    expected_pairs = len(scenes) * len(frame_indices)
    return {
        "format_version": 1,
        "config": str(config_path),
        "config_sha256": _sha256_and_crc32(config_path)[0],
        "output": str(output),
        "expected_scenes": len(scenes),
        "completed_scenes": len(completed_scene_names),
        "expected_pairs": expected_pairs,
        "completed_pairs": completed_pairs,
        "completion_percent": 100.0 * completed_pairs / expected_pairs,
        "partial_files": partials,
        "errors": errors,
        "status": (
            "complete"
            if completed_pairs == expected_pairs and not errors and not partials
            else "in_progress"
        ),
    }
