"""项目统一命令行入口。

命令覆盖模型统计、结构化剪枝检查、数据生成/导入、清单验证、训练、
ONNX 导出与算子审计；自动化脚本应调用这些稳定子命令而不是复制内部逻辑。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isp_ai_enhancement.data.manifest import read_manifest, validate_manifest
from isp_ai_enhancement.data.sidd import import_sidd_dataset
from isp_ai_enhancement.data.synthetic import generate_smoke_dataset
from isp_ai_enhancement.export import export_onnx
from isp_ai_enhancement.models.factory import build_model_from_file
from isp_ai_enhancement.onnx_audit import audit_onnx
from isp_ai_enhancement.pruning.physical import physical_prune
from isp_ai_enhancement.pruning.torch_pruning_adapter import (
    torch_pruning_physical_prune,
)
from isp_ai_enhancement.training.engine import train_from_config


def _model_summary(args: argparse.Namespace) -> int:
    """打印模型通道、扩展规格和参数量的机器可读 JSON 摘要。"""

    model = build_model_from_file(args.config)
    print(
        json.dumps(
            {
                "parameters": model.parameter_count(),
                "trainable_parameters": model.parameter_count(trainable_only=True),
                "input_channels": model.input_channels,
                "output_channels": model.output_channels,
                "width": model.width,
                "expansion_spec": model.expansion_spec.as_dict(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _pruning_summary(args: argparse.Namespace) -> int:
    """执行选定结构化剪枝后端并报告真实参数缩减和依赖图统计。"""

    source = build_model_from_file(args.source)
    target_config = build_model_from_file(args.target)
    backend_details = None
    if args.backend == "torch-pruning":
        _target, report, backend = torch_pruning_physical_prune(
            source, target_config.expansion_spec
        )
        backend_details = {
            "version": backend.backend_version,
            "dependency_groups": backend.dependency_groups,
            "dependency_operations": backend.dependency_operations,
            "pruned_gate_units": backend.pruned_gate_units,
        }
    else:
        _target, report = physical_prune(source, target_config.expansion_spec)
    print(
        json.dumps(
            {
                "backend": args.backend,
                "backend_details": backend_details,
                "source_parameters": report.source_parameters,
                "target_parameters": report.target_parameters,
                "physical_pruning_ratio": report.pruning_ratio,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _make_smoke_data(args: argparse.Namespace) -> int:
    """生成小规模合成数据，用于验证端到端流水线是否可运行。"""

    manifest = generate_smoke_dataset(
        args.output,
        samples=args.samples,
        height=args.height,
        width=args.width,
        seed=args.seed,
    )
    print(manifest)
    return 0


def _validate_manifest(args: argparse.Namespace) -> int:
    """校验清单及其引用文件，发现问题时返回非零退出码。"""

    manifest = Path(args.manifest)
    errors = validate_manifest(read_manifest(manifest), root=manifest.parent)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"OK: {manifest}")
    return 0


def _import_sidd(args: argparse.Namespace) -> int:
    """把官方 SIDD 场景目录转换为项目统一 packed RAW 格式。"""

    print(
        import_sidd_dataset(
            args.source,
            args.output,
            nlf_csv=args.nlf_csv,
            split_seed=args.split_seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
    )
    return 0


def _train(args: argparse.Namespace) -> int:
    """调用配置驱动训练入口并打印最终检查点位置。"""

    print(train_from_config(args.config))
    return 0


def _export_onnx(args: argparse.Namespace) -> int:
    """导出固定输入契约的 ONNX，并打印产物位置。"""

    print(
        export_onnx(
            model_config=args.config,
            checkpoint=args.checkpoint,
            output=args.output,
            export_config=args.export_config,
        )
    )
    return 0


def _audit_onnx(args: argparse.Namespace) -> int:
    """审计 ONNX 算子与输入输出形状，打印 JSON 报告。"""

    print(json.dumps(audit_onnx(args.model), indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """构建所有子命令和参数定义；调用本函数不会解析或修改进程状态。"""

    parser = argparse.ArgumentParser(prog="isp-ai")
    commands = parser.add_subparsers(dest="command", required=True)

    summary = commands.add_parser("model-summary")
    summary.add_argument("--config", required=True)
    summary.set_defaults(function=_model_summary)

    pruning = commands.add_parser("pruning-summary")
    pruning.add_argument("--source", required=True)
    pruning.add_argument("--target", required=True)
    pruning.add_argument(
        "--backend",
        choices=("manual", "torch-pruning"),
        default="torch-pruning",
    )
    pruning.set_defaults(function=_pruning_summary)

    smoke = commands.add_parser("make-smoke-data")
    smoke.add_argument("--output", required=True)
    smoke.add_argument("--samples", type=int, default=16)
    smoke.add_argument("--height", type=int, default=96)
    smoke.add_argument("--width", type=int, default=96)
    smoke.add_argument("--seed", type=int, default=20260726)
    smoke.set_defaults(function=_make_smoke_data)

    manifest = commands.add_parser("validate-manifest")
    manifest.add_argument("--manifest", required=True)
    manifest.set_defaults(function=_validate_manifest)

    sidd = commands.add_parser("import-sidd")
    sidd.add_argument("--source", required=True)
    sidd.add_argument("--output", required=True)
    sidd.add_argument("--nlf-csv")
    sidd.add_argument("--split-seed", type=int, default=20260726)
    sidd.add_argument("--train-ratio", type=float, default=0.8)
    sidd.add_argument("--val-ratio", type=float, default=0.1)
    sidd.set_defaults(function=_import_sidd)

    train = commands.add_parser("train")
    train.add_argument("--config", required=True)
    train.set_defaults(function=_train)

    onnx = commands.add_parser("export-onnx")
    onnx.add_argument("--config", required=True)
    onnx.add_argument("--checkpoint", required=True)
    onnx.add_argument("--output", required=True)
    onnx.add_argument("--export-config", default="configs/export_onnx.yaml")
    onnx.set_defaults(function=_export_onnx)

    audit = commands.add_parser("audit-onnx")
    audit.add_argument("--model", required=True)
    audit.set_defaults(function=_audit_onnx)
    return parser


def main() -> int:
    """解析命令行并执行已绑定处理函数，返回适合 shell 的退出码。"""

    args = build_parser().parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
