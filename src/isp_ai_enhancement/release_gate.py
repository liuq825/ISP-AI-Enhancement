"""把商用级画质、性能、稳定性和交付证据转换为机器可执行放行结论。

本模块不生成或猜测真机数据。证据缺失时返回 ``BLOCKED``，指标明确越界时返回
``FAIL``，只有全部 Gate 都具有真实输入且通过时才返回 ``PASS``。这种三态语义可防止
“尚未测试”被误写成“测试通过”，也不允许某个 Sensor 的失败被总体平均值抵消。
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from isp_ai_enhancement.config import load_yaml

_SHA256_LENGTH = 64


class MissingEvidence(ValueError):
    """表示放行所需字段或文件证据尚未提供，而不是指标已经测试失败。"""


@dataclass(frozen=True)
class GateResult:
    """保存单个 Gate 的三态结果和可操作原因。"""

    name: str
    status: str
    reasons: tuple[str, ...]


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    """读取嵌套 mapping；缺失或类型错误统一归类为证据不足。"""

    if not isinstance(value, Mapping):
        raise MissingEvidence(f"{field} 必须是 mapping")
    return value


def _required(mapping: Mapping[str, Any], key: str, field: str) -> Any:
    """读取必填字段，并拒绝空字符串和显式空值。"""

    if key not in mapping or mapping[key] is None:
        raise MissingEvidence(f"缺少 {field}.{key}")
    value = mapping[key]
    if isinstance(value, str) and not value.strip():
        raise MissingEvidence(f"{field}.{key} 不能为空")
    return value


def _number(mapping: Mapping[str, Any], key: str, field: str) -> float:
    """读取有限浮点数，拒绝布尔值、NaN 与无穷大。"""

    value = _required(mapping, key, field)
    if isinstance(value, bool):
        raise MissingEvidence(f"{field}.{key} 必须是有限数值")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise MissingEvidence(f"{field}.{key} 必须是有限数值") from error
    if not math.isfinite(result):
        raise MissingEvidence(f"{field}.{key} 必须是有限数值")
    return result


def _integer(mapping: Mapping[str, Any], key: str, field: str) -> int:
    """读取整数计数，拒绝以浮点截断方式掩盖不合法证据。"""

    value = _required(mapping, key, field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise MissingEvidence(f"{field}.{key} 必须是整数")
    return value


def _boolean(mapping: Mapping[str, Any], key: str, field: str) -> bool:
    """读取严格布尔字段，不接受容易误解的字符串 ``yes/no``。"""

    value = _required(mapping, key, field)
    if not isinstance(value, bool):
        raise MissingEvidence(f"{field}.{key} 必须是布尔值")
    return value


def _sha256(path: Path) -> str:
    """分块计算文件 SHA256，避免模型或报告整体进入内存。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(path: Path) -> str:
    """返回证据 YAML 自身的 SHA256，绑定本次 Gate 输入版本。"""

    return _sha256(path)


def _valid_sha256(value: Any) -> bool:
    """判断值是否为非占位的 64 位十六进制 SHA256。"""

    text = str(value)
    return (
        len(text) == _SHA256_LENGTH
        and any(character != "0" for character in text)
        and all(character in "0123456789abcdefABCDEF" for character in text)
    )


def _gate(name: str, checker: Callable[[], list[str]]) -> GateResult:
    """执行单项检查，把缺失证据与真实阈值失败转换为稳定三态结果。"""

    try:
        reasons = checker()
    except MissingEvidence as error:
        return GateResult(name=name, status="BLOCKED", reasons=(str(error),))
    if reasons:
        return GateResult(name=name, status="FAIL", reasons=tuple(reasons))
    return GateResult(name=name, status="PASS", reasons=())


def _check_target(root: Mapping[str, Any]) -> list[str]:
    """确认报告确实来自麒麟 9000 目标机型及锁定的软硬件版本。"""

    format_version = _integer(root, "format_version", "root")
    _required(root, "release_id", "root")
    target = _mapping(_required(root, "target", "root"), "target")
    chip = str(_required(target, "chip", "target")).strip().lower().replace(" ", "")
    reasons = []
    if format_version != 1:
        reasons.append(f"root.format_version={format_version}，当前仅支持 1")
    if chip not in {"kirin9000", "麒麟9000"}:
        reasons.append(f"target.chip={chip!r} 不是麒麟 9000")
    for key in ("device_model", "os_build", "ddk_version", "firmware_version"):
        _required(target, key, "target")
    return reasons


def _check_data(root: Mapping[str, Any]) -> list[str]:
    """检查数据冻结、来源审阅、切分、真值与 Golden Set 哈希证据。"""

    data = _mapping(_required(root, "data", "root"), "data")
    reasons = []
    for key in (
        "usage_reviewed",
        "split_validated",
        "ground_truth_validated",
        "metadata_validated",
        "golden_set_frozen",
    ):
        if not _boolean(data, key, "data"):
            reasons.append(f"data.{key} 未通过")
    for key in ("manifest_sha256", "golden_set_sha256"):
        if not _valid_sha256(_required(data, key, "data")):
            reasons.append(f"data.{key} 不是合法 SHA256")
    return reasons


def _quality_domains(root: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], float]:
    """读取逐 Sensor×模式指标，禁止使用单一总体平均值代替域级结果。"""

    quality = _mapping(_required(root, "quality", "root"), "quality")
    raw_domains = _required(quality, "domains", "quality")
    if not isinstance(raw_domains, list) or not raw_domains:
        raise MissingEvidence("quality.domains 必须包含至少一个 Sensor×模式")
    domains = [
        _mapping(value, f"quality.domains[{index}]")
        for index, value in enumerate(raw_domains)
    ]
    max_ssim_drop = _number(quality, "max_ssim_drop", "quality")
    if max_ssim_drop < 0:
        raise MissingEvidence("quality.max_ssim_drop 不能为负数")
    return domains, max_ssim_drop


def _domain_name(domain: Mapping[str, Any], index: int) -> str:
    """构造稳定域名称，便于失败报告直接定位 Sensor 和工作模式。"""

    sensor = str(_required(domain, "sensor_id", f"quality.domains[{index}]"))
    mode = str(_required(domain, "mode", f"quality.domains[{index}]"))
    return f"{sensor}×{mode}"


def _check_p0(root: Mapping[str, Any]) -> list[str]:
    """要求每个域都单独通过 P0 FP32 画质验收。"""

    domains, _max_ssim_drop = _quality_domains(root)
    reasons = []
    for index, domain in enumerate(domains):
        name = _domain_name(domain, index)
        if not _boolean(domain, "p0_pass", f"quality.domains[{index}]"):
            reasons.append(f"{name}: P0 FP32 未通过")
        p0_psnr = _number(domain, "p0_psnr_db", f"quality.domains[{index}]")
        p0_ssim = _number(domain, "p0_ssim", f"quality.domains[{index}]")
        if p0_psnr <= 0:
            reasons.append(f"{name}: P0 PSNR 必须大于 0 dB")
        if not 0 <= p0_ssim <= 1:
            reasons.append(f"{name}: P0 SSIM 必须在 [0,1]")
    return reasons


def _check_pruning(root: Mapping[str, Any]) -> list[str]:
    """检查每个域的物理剪枝 FP32 相对 P0 下降不超过 0.10 dB。"""

    domains, _max_ssim_drop = _quality_domains(root)
    reasons = []
    for index, domain in enumerate(domains):
        name = _domain_name(domain, index)
        p0 = _number(domain, "p0_psnr_db", f"quality.domains[{index}]")
        pruned = _number(domain, "pruned_fp32_psnr_db", f"quality.domains[{index}]")
        drop = p0 - pruned
        if pruned <= 0:
            reasons.append(f"{name}: 剪枝 FP32 PSNR 必须大于 0 dB")
        if drop > 0.10 + 1e-9:
            reasons.append(f"{name}: 剪枝 PSNR 下降 {drop:.4f} dB > 0.10 dB")
    return reasons


def _check_qat(root: Mapping[str, Any]) -> list[str]:
    """检查目标 OM 相对剪枝 FP32 的逐域下降不超过 0.08 dB。"""

    domains, _max_ssim_drop = _quality_domains(root)
    reasons = []
    for index, domain in enumerate(domains):
        name = _domain_name(domain, index)
        pruned = _number(domain, "pruned_fp32_psnr_db", f"quality.domains[{index}]")
        final = _number(domain, "final_om_psnr_db", f"quality.domains[{index}]")
        drop = pruned - final
        if final <= 0:
            reasons.append(f"{name}: 最终 OM PSNR 必须大于 0 dB")
        if drop > 0.08 + 1e-9:
            reasons.append(f"{name}: QAT/OM PSNR 下降 {drop:.4f} dB > 0.08 dB")
    return reasons


def _check_final_quality(root: Mapping[str, Any]) -> list[str]:
    """检查最终 OM 的 P0 总下降、SSIM、尾部和盲评逐域结论。"""

    domains, max_ssim_drop = _quality_domains(root)
    reasons = []
    for index, domain in enumerate(domains):
        field = f"quality.domains[{index}]"
        name = _domain_name(domain, index)
        p0_psnr = _number(domain, "p0_psnr_db", field)
        final_psnr = _number(domain, "final_om_psnr_db", field)
        p0_ssim = _number(domain, "p0_ssim", field)
        final_ssim = _number(domain, "final_om_ssim", field)
        if not 0 <= final_ssim <= 1:
            reasons.append(f"{name}: 最终 OM SSIM 必须在 [0,1]")
        if p0_psnr - final_psnr > 0.15 + 1e-9:
            reasons.append(
                f"{name}: 最终 PSNR 相对 P0 下降 {p0_psnr - final_psnr:.4f} dB > 0.15 dB"
            )
        if p0_ssim - final_ssim > max_ssim_drop + 1e-12:
            reasons.append(
                f"{name}: 最终 SSIM 下降 {p0_ssim - final_ssim:.6f} "
                f"> {max_ssim_drop:.6f}"
            )
        if not _boolean(domain, "tail_pass", field):
            reasons.append(f"{name}: 尾部样本未通过")
        if not _boolean(domain, "blind_review_pass", field):
            reasons.append(f"{name}: 盲评未通过")
    return reasons


def _check_operator(root: Mapping[str, Any]) -> list[str]:
    """检查目标 profiler 的 INT8 计算量占比及未解释回退节点。"""

    operator = _mapping(_required(root, "operator", "root"), "operator")
    ratio = _number(operator, "int8_mac_ratio", "operator")
    reasons = []
    if not 0 <= ratio <= 1:
        reasons.append(f"INT8 MAC 占比 {ratio} 必须在 [0,1]")
    elif ratio < 0.95 - 1e-12:
        reasons.append(f"INT8 MAC 占比 {ratio:.4%} < 95%")
    for key in ("unexplained_fallback_nodes", "unexplained_cast_nodes"):
        count = _integer(operator, key, "operator")
        if count < 0:
            reasons.append(f"operator.{key} 不能为负数")
        elif count != 0:
            reasons.append(f"operator.{key}={count}，要求为 0")
    if not _valid_sha256(_required(operator, "profiler_report_sha256", "operator")):
        reasons.append("operator.profiler_report_sha256 不是合法 SHA256")
    return reasons


def _check_performance(root: Mapping[str, Any]) -> list[str]:
    """检查热态样本量、P90 150 ms 上限和产品自定义 P99 门限。"""

    performance = _mapping(_required(root, "performance", "root"), "performance")
    sample_count = _integer(performance, "hot_sample_count", "performance")
    p90 = _number(performance, "hot_p90_ms", "performance")
    p99 = _number(performance, "hot_p99_ms", "performance")
    p99_limit = _number(performance, "p99_limit_ms", "performance")
    reasons = []
    if sample_count < 0:
        reasons.append("热态样本数不能为负数")
    elif sample_count < 100:
        reasons.append(f"热态样本数 {sample_count} < 100")
    if min(p90, p99, p99_limit) < 0:
        reasons.append("热态时延和 P99 门限不能为负数")
    if p90 > 150.0 + 1e-9:
        reasons.append(f"热态 P90 {p90:.3f} ms > 150 ms")
    if p99 > p99_limit + 1e-9:
        reasons.append(f"热态 P99 {p99:.3f} ms > 产品门限 {p99_limit:.3f} ms")
    if p99 + 1e-9 < p90:
        reasons.append(f"热态 P99 {p99:.3f} ms 小于 P90 {p90:.3f} ms，统计口径无效")
    return reasons


def _check_memory(root: Mapping[str, Any]) -> list[str]:
    """检查融合缓存释放后的 AI 峰值内存是否满足产品预算。"""

    memory = _mapping(_required(root, "memory", "root"), "memory")
    peak = _number(memory, "ai_peak_mb", "memory")
    budget = _number(memory, "budget_mb", "memory")
    reasons = []
    if peak < 0:
        reasons.append("memory.ai_peak_mb 不能为负数")
    if budget <= 0:
        reasons.append("memory.budget_mb 必须大于 0")
    if not _boolean(memory, "fusion_cache_released", "memory"):
        reasons.append("内存测试时尚未释放融合缓存，测量口径无效")
    if peak > budget + 1e-9:
        reasons.append(f"AI 峰值 {peak:.3f} MB > 预算 {budget:.3f} MB")
    return reasons


def _check_stability(root: Mapping[str, Any]) -> list[str]:
    """检查长稳、连拍、切摄、生命周期和异常输入场景均无故障。"""

    stability = _mapping(_required(root, "stability", "root"), "stability")
    reasons = []
    for key in (
        "long_run_pass",
        "burst20_pass",
        "camera_switch_pass",
        "lifecycle_pass",
        "invalid_input_pass",
    ):
        if not _boolean(stability, key, "stability"):
            reasons.append(f"stability.{key} 未通过")
    for key in ("crash_count", "timeout_count"):
        count = _integer(stability, key, "stability")
        if count < 0:
            reasons.append(f"stability.{key} 不能为负数")
        elif count != 0:
            reasons.append(f"stability.{key}={count}，要求为 0")
    return reasons


def _check_fallback(root: Mapping[str, Any]) -> list[str]:
    """检查故障注入必须 100% 出图，并完整记录稳定 reason code。"""

    fallback = _mapping(_required(root, "fallback", "root"), "fallback")
    trials = _integer(fallback, "fault_injection_trials", "fallback")
    successes = _integer(fallback, "successful_outputs", "fallback")
    reasons = []
    if trials <= 0:
        reasons.append("fallback.fault_injection_trials 必须大于 0")
    if successes < 0:
        reasons.append("fallback.successful_outputs 不能为负数")
    elif trials > 0 and successes != trials:
        reasons.append(f"故障注入出图 {successes}/{trials}，要求 100%")
    if not _boolean(fallback, "reason_codes_complete", "fallback"):
        reasons.append("回退 reason code 不完整")
    return reasons


def _resolve_artifact(evidence_path: Path, value: Any) -> Path:
    """把交付证据中的相对路径解析到证据文件所在目录。"""

    path = Path(str(value))
    return path if path.is_absolute() else evidence_path.parent / path


def _check_release(root: Mapping[str, Any], evidence_path: Path) -> list[str]:
    """核验交付文件真实存在、哈希匹配，并具有明确签字人。"""

    release = _mapping(_required(root, "release", "root"), "release")
    reasons = []
    for prefix in ("model_card", "model_manifest", "om_model", "calibration", "compatibility"):
        path_value = _required(release, f"{prefix}_path", "release")
        expected = _required(release, f"{prefix}_sha256", "release")
        if not _valid_sha256(expected):
            reasons.append(f"release.{prefix}_sha256 不是合法 SHA256")
            continue
        path = _resolve_artifact(evidence_path, path_value)
        if not path.is_file():
            reasons.append(f"交付文件不存在：{path}")
            continue
        actual = _sha256(path)
        if actual.lower() != str(expected).lower():
            reasons.append(f"{path} SHA256 不匹配：{actual}")
    signers = _required(release, "signoff", "release")
    if not isinstance(signers, list) or not signers or not all(
        isinstance(value, str) and value.strip() for value in signers
    ):
        raise MissingEvidence("release.signoff 必须包含至少一名明确签字人")
    return reasons


def evaluate_release_evidence(path: str | Path) -> dict[str, Any]:
    """评估一份 YAML 证据并返回可序列化、可审计的完整 Gate 报告。"""

    evidence_path = Path(path)
    root = load_yaml(evidence_path)
    gates = [
        _gate("target", lambda: _check_target(root)),
        _gate("data", lambda: _check_data(root)),
        _gate("p0", lambda: _check_p0(root)),
        _gate("pruning", lambda: _check_pruning(root)),
        _gate("qat_om", lambda: _check_qat(root)),
        _gate("final_quality", lambda: _check_final_quality(root)),
        _gate("operator", lambda: _check_operator(root)),
        _gate("performance", lambda: _check_performance(root)),
        _gate("memory", lambda: _check_memory(root)),
        _gate("stability", lambda: _check_stability(root)),
        _gate("fallback", lambda: _check_fallback(root)),
        _gate("release", lambda: _check_release(root, evidence_path)),
    ]
    statuses = {gate.status for gate in gates}
    overall = "FAIL" if "FAIL" in statuses else "BLOCKED" if "BLOCKED" in statuses else "PASS"
    return {
        "format_version": 1,
        "release_id": str(root.get("release_id", "")),
        "evidence_path": str(evidence_path.resolve()),
        "evidence_sha256": _sha256_text(evidence_path),
        "overall_status": overall,
        "gates": [asdict(gate) for gate in gates],
    }


def write_release_report(report: Mapping[str, Any], output: str | Path) -> Path:
    """以同目录临时文件原子写入 JSON 报告，避免中断留下半截放行结论。"""

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination
