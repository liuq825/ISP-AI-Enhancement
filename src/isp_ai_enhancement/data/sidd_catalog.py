"""从 SIDD 官方场景页和两份镜像清单生成版本化 Range 获取配置。

官网另行发布的 Mirror 1/2 文本均按每个非 held-out 场景依次列出 noisy RAW、
GT RAW、noisy sRGB、GT sRGB 和 metadata 五个 URL。本模块逐 HTML 表格行提取
场景身份，再按严格的 ``场景数×5`` 契约分别绑定两份 URL，避免跨行正则造成
静默错配；运行时可在主镜像连接失败后切换到备用镜像。
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Sequence
from datetime import date
from html.parser import HTMLParser
from pathlib import Path

import yaml

from .sidd import SIDDScene

SIDD_SCENE_PAGE = "https://abdokamel.github.io/sidd/dataset.html"
SIDD_MIRROR1_LIST = "https://abdokamel.github.io/sidd/files/SIDD_URLs.txt"
SIDD_MIRROR2_LIST = "https://abdokamel.github.io/sidd/files/SIDD_URLs_Mirror_2.txt"

# 只在单个已隔离表格行内搜索完整场景令牌；相机与末尾亮度字段沿用官方枚举。
_SCENE_TOKEN = re.compile(
    r"\b\d{4}_\d{3}_(?:GP|IP|S6|N6|G4)_\d{5}_\d{5}_\d{4}_[LNH]\b",
    re.IGNORECASE,
)
TextFetcher = Callable[[str], bytes]


class _TableRowTextParser(HTMLParser):
    """只收集每个 HTML ``tr`` 内的可见文本，保持行边界不可跨越。"""

    def __init__(self) -> None:
        """初始化行状态；标准实体会由父类自动转换为 Unicode。"""

        super().__init__(convert_charrefs=True)
        self.rows: list[str] = []
        self._inside_row = False
        self._parts: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        _attrs: list[tuple[str, str | None]],
    ) -> None:
        """遇到表格行起点时创建独立缓冲区，禁止复用上一行文本。"""

        if tag.casefold() == "tr":
            self._inside_row = True
            self._parts = []

    def handle_data(self, data: str) -> None:
        """仅在表格行内部保留非空文本片段。"""

        if self._inside_row and data.strip():
            self._parts.append(data.strip())

    def handle_endtag(self, tag: str) -> None:
        """行结束时固化文本并立即清空状态。"""

        if tag.casefold() == "tr" and self._inside_row:
            self.rows.append(" ".join(self._parts))
            self._inside_row = False
            self._parts = []


def _canonical_scene_name(value: str) -> str:
    """严格解析场景令牌并返回与导入器一致的规范目录名。"""

    scene = SIDDScene.from_directory(Path(value))
    return (
        f"{scene.instance_id}_{scene.scene_id}_{scene.camera_id}_{scene.iso:05d}_"
        f"{scene.shutter_denominator:05d}_{scene.cct:04d}_{scene.brightness}"
    )


def _training_scene_names(page_content: bytes) -> tuple[list[str], int]:
    """按 HTML 行顺序返回非 held-out 场景，并统计被排除的 benchmark 行。"""

    parser = _TableRowTextParser()
    parser.feed(page_content.decode("utf-8"))
    scenes: list[str] = []
    held_out_count = 0
    seen: set[str] = set()
    for row in parser.rows:
        match = _SCENE_TOKEN.search(row)
        if match is None:
            continue
        scene_name = _canonical_scene_name(match.group())
        if scene_name in seen:
            raise ValueError(f"官方场景表出现重复场景：{scene_name}")
        seen.add(scene_name)
        if "held for benchmark" in row.casefold():
            held_out_count += 1
        else:
            scenes.append(scene_name)
    if not scenes:
        raise ValueError("官方场景页未解析出任何训练场景")
    return scenes, held_out_count


def _default_fetcher(url: str) -> bytes:
    """以显式超时读取官方文本；网络依赖只在实际生成配置时加载。"""

    try:
        import requests
    except ImportError as error:
        raise RuntimeError(
            "生成 SIDD 获取配置需要 acquisition 依赖："
            "pip install -e '.[acquisition]'"
        ) from error
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return bytes(response.content)


def _validated_frames(frame_indices: Sequence[int]) -> list[int]:
    """校验并保留帧顺序，拒绝 bool、越界值和重复项。"""

    frames = list(frame_indices)
    if not frames or any(
        not isinstance(frame, int)
        or isinstance(frame, bool)
        or not 1 <= frame <= 999
        for frame in frames
    ):
        raise ValueError("frame_indices 必须是 [1,999] 内的非空整数序列")
    if len(frames) != len(set(frames)):
        raise ValueError("frame_indices 含重复帧")
    return frames


def _validated_mirror_groups(
    *,
    content: bytes,
    source_name: str,
    scene_count: int,
) -> list[list[str]]:
    """把一份官方镜像清单严格拆成逐场景五角色 URL 组。

    Mirror 1 使用 HTTP/IP，Mirror 2 使用 HTTPS/CodaLab，因此这里只要求 HTTP(S)
    且不把传输协议当作数据身份。成员名、CRC 和 SHA256 仍由下载器逐层校验。
    """

    urls = content.decode("utf-8").split()
    expected_urls = scene_count * 5
    if len(urls) != expected_urls:
        raise ValueError(f"{source_name} URL 数应为 {expected_urls}，实际 {len(urls)}")
    if len(urls) != len(set(urls)):
        raise ValueError(f"{source_name} 清单含重复 URL")
    if any(not url.startswith(("http://", "https://")) for url in urls):
        raise ValueError(f"{source_name} 清单必须全部使用 HTTP(S) URL")
    return [
        urls[index * 5 : (index + 1) * 5]
        for index in range(scene_count)
    ]


def build_sidd_range_config(
    *,
    output: str | Path,
    frame_indices: Sequence[int] = (10, 20),
    source_page: str = SIDD_SCENE_PAGE,
    mirror_list: str = SIDD_MIRROR2_LIST,
    fallback_mirror_list: str | None = SIDD_MIRROR1_LIST,
    expected_training_scenes: int = 160,
    fetcher: TextFetcher | None = None,
) -> Path:
    """生成带可选备用镜像的非 held-out SIDD 双帧 Range 获取 YAML。

    场景页、主镜像清单和可选备用清单均完整下载并计算 SHA256。每份镜像对每个
    训练场景必须恰有五个 HTTP(S) URL，且场景页训练数必须等于预期值；任一数量
    变化都拒绝生成配置。输出只保留 noisy/GT 两个 RAW URL，sRGB 与 metadata
    URL 不进入训练获取队列。
    """

    if expected_training_scenes <= 0:
        raise ValueError("expected_training_scenes 必须为正整数")
    frames = _validated_frames(frame_indices)
    loader = fetcher or _default_fetcher
    page_content = loader(source_page)
    mirror_content = loader(mirror_list)
    fallback_content = (
        loader(fallback_mirror_list)
        if fallback_mirror_list is not None
        else None
    )
    scenes, held_out_count = _training_scene_names(page_content)
    if len(scenes) != expected_training_scenes:
        raise ValueError(
            f"官方训练场景数应为 {expected_training_scenes}，实际 {len(scenes)}"
        )
    primary_groups = _validated_mirror_groups(
        content=mirror_content,
        source_name="主镜像",
        scene_count=len(scenes),
    )
    fallback_groups = (
        _validated_mirror_groups(
            content=fallback_content,
            source_name="备用镜像",
            scene_count=len(scenes),
        )
        if fallback_content is not None
        else None
    )

    entries: list[dict[str, str]] = []
    for index, scene_name in enumerate(scenes):
        primary = primary_groups[index]
        entry = {
            "scene": scene_name,
            "noisy_url": primary[0],
            "ground_truth_url": primary[1],
        }
        if fallback_groups is not None:
            fallback = fallback_groups[index]
            entry["fallback_noisy_url"] = fallback[0]
            entry["fallback_ground_truth_url"] = fallback[1]
        entries.append(entry)
    config = {
        "source_page": source_page,
        "source_page_sha256": hashlib.sha256(page_content).hexdigest(),
        "mirror_list": mirror_list,
        "mirror_list_sha256": hashlib.sha256(mirror_content).hexdigest(),
        "checked_on": date.today().isoformat(),
        "held_out_scene_count": held_out_count,
        "frame_indices": frames,
        "scenes": entries,
    }
    if fallback_mirror_list is not None and fallback_content is not None:
        config["fallback_mirror_list"] = fallback_mirror_list
        config["fallback_mirror_list_sha256"] = hashlib.sha256(
            fallback_content
        ).hexdigest()
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp")
    temporary.write_text(
        "# 由 build-sidd-range-config 从两份官方来源生成；人工修改后必须重新核验哈希。\n"
        + yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(destination)
    return destination
