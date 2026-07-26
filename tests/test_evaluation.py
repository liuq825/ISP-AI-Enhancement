"""验证治理门禁后的基线评测、分桶统计与报告落盘。"""

from pathlib import Path

from isp_ai_enhancement.data.synthetic import generate_smoke_dataset
from isp_ai_enhancement.evaluation import evaluate_manifest


def test_evaluate_manifest_reports_noisy_baseline_without_model(tmp_path: Path) -> None:
    """没有 checkpoint 时仍应输出可追溯 noisy PSNR 和 Sensor/ISO 分桶。"""

    manifest = generate_smoke_dataset(
        tmp_path / "smoke",
        samples=8,
        height=32,
        width=32,
    )
    output = tmp_path / "baseline.json"
    report = evaluate_manifest(
        manifest=manifest,
        split="val",
        context_config="configs/context.yaml",
        catalog="resources/datasets.yaml",
        purpose="smoke",
        output=output,
        batch_size=2,
    )
    assert report["overall"]["samples"] == 1
    assert report["overall"]["noisy_psnr_db"] > 0
    assert "enhanced_psnr_db" not in report["overall"]
    assert set(report["by_sensor"]) == {"smoke_sensor"}
    assert report["model"] is None
    assert output.is_file()
