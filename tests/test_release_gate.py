"""验证商用级 Gate 的三态语义、逐域阈值和交付文件哈希。"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

from isp_ai_enhancement.release_gate import (
    evaluate_release_evidence,
    write_release_report,
)


def _sha256(path: Path) -> str:
    """计算测试交付文件的 SHA256。"""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _passing_evidence(tmp_path: Path) -> dict[str, Any]:
    """构造覆盖全部 Gate 的最小通过证据和真实哈希文件。"""

    release: dict[str, Any] = {"signoff": ["画质负责人", "端侧负责人"]}
    for prefix in ("model_card", "model_manifest", "om_model", "calibration", "compatibility"):
        artifact = tmp_path / f"{prefix}.bin"
        artifact.write_bytes(f"verified-{prefix}".encode())
        release[f"{prefix}_path"] = artifact.name
        release[f"{prefix}_sha256"] = _sha256(artifact)
    digest = "a" * 64
    return {
        "format_version": 1,
        "release_id": "unit-test",
        "target": {
            "chip": "Kirin 9000",
            "device_model": "test-device",
            "os_build": "test-os",
            "ddk_version": "test-ddk",
            "firmware_version": "test-firmware",
        },
        "data": {
            "usage_reviewed": True,
            "split_validated": True,
            "ground_truth_validated": True,
            "metadata_validated": True,
            "golden_set_frozen": True,
            "manifest_sha256": digest,
            "golden_set_sha256": digest,
        },
        "quality": {
            "max_ssim_drop": 0.002,
            "domains": [
                {
                    "sensor_id": "sensor-a",
                    "mode": "single",
                    "p0_pass": True,
                    "p0_psnr_db": 40.0,
                    "p0_ssim": 0.990,
                    "pruned_fp32_psnr_db": 39.92,
                    "final_om_psnr_db": 39.85,
                    "final_om_ssim": 0.989,
                    "tail_pass": True,
                    "blind_review_pass": True,
                }
            ],
        },
        "operator": {
            "int8_mac_ratio": 0.96,
            "unexplained_fallback_nodes": 0,
            "unexplained_cast_nodes": 0,
            "profiler_report_sha256": digest,
        },
        "performance": {
            "hot_sample_count": 100,
            "hot_p90_ms": 149.0,
            "hot_p99_ms": 170.0,
            "p99_limit_ms": 180.0,
        },
        "memory": {
            "fusion_cache_released": True,
            "ai_peak_mb": 255.0,
            "budget_mb": 256.0,
        },
        "stability": {
            "long_run_pass": True,
            "burst20_pass": True,
            "camera_switch_pass": True,
            "lifecycle_pass": True,
            "invalid_input_pass": True,
            "crash_count": 0,
            "timeout_count": 0,
        },
        "fallback": {
            "fault_injection_trials": 20,
            "successful_outputs": 20,
            "reason_codes_complete": True,
        },
        "release": release,
    }


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    """以 UTF-8 写入测试 YAML，保留中文签字人。"""

    path.write_text(
        yaml.safe_dump(value, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def test_complete_release_evidence_passes_and_writes_atomic_report(tmp_path: Path) -> None:
    """完整真实证据应全部通过，并可写出绑定输入哈希的 JSON 报告。"""

    evidence_path = tmp_path / "release.yaml"
    _write_yaml(evidence_path, _passing_evidence(tmp_path))
    report = evaluate_release_evidence(evidence_path)
    assert report["overall_status"] == "PASS"
    assert {gate["status"] for gate in report["gates"]} == {"PASS"}
    output = write_release_report(report, tmp_path / "report.json")
    assert output.is_file()
    assert not (tmp_path / "report.json.tmp").exists()


def test_one_domain_failure_cannot_be_hidden_by_other_domain(tmp_path: Path) -> None:
    """任一 Sensor×模式超阈值都必须失败，不能被另一个高分域平均抵消。"""

    evidence = _passing_evidence(tmp_path)
    failing = dict(evidence["quality"]["domains"][0])
    failing["sensor_id"] = "sensor-b"
    failing["final_om_psnr_db"] = 39.60
    evidence["quality"]["domains"].append(failing)
    evidence_path = tmp_path / "release.yaml"
    _write_yaml(evidence_path, evidence)

    report = evaluate_release_evidence(evidence_path)
    assert report["overall_status"] == "FAIL"
    failures = {
        gate["name"]: gate["reasons"]
        for gate in report["gates"]
        if gate["status"] == "FAIL"
    }
    assert "sensor-b×single" in "\n".join(failures["qat_om"])
    assert "sensor-b×single" in "\n".join(failures["final_quality"])


def test_missing_target_evidence_is_blocked_not_passed(tmp_path: Path) -> None:
    """未填写目标设备版本属于阻塞，不能静默视为通过。"""

    evidence = _passing_evidence(tmp_path)
    evidence["target"]["ddk_version"] = ""
    evidence_path = tmp_path / "release.yaml"
    _write_yaml(evidence_path, evidence)

    report = evaluate_release_evidence(evidence_path)
    target = next(gate for gate in report["gates"] if gate["name"] == "target")
    assert target["status"] == "BLOCKED"
    assert report["overall_status"] == "BLOCKED"
