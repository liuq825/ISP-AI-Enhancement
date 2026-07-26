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
from isp_ai_enhancement.training.engine import train_from_config


def _model_summary(args: argparse.Namespace) -> int:
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
    source = build_model_from_file(args.source)
    target_config = build_model_from_file(args.target)
    _target, report = physical_prune(source, target_config.expansion_spec)
    print(
        json.dumps(
            {
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
    manifest = Path(args.manifest)
    errors = validate_manifest(read_manifest(manifest), root=manifest.parent)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"OK: {manifest}")
    return 0


def _import_sidd(args: argparse.Namespace) -> int:
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
    print(train_from_config(args.config))
    return 0


def _export_onnx(args: argparse.Namespace) -> int:
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
    print(json.dumps(audit_onnx(args.model), indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="isp-ai")
    commands = parser.add_subparsers(dest="command", required=True)

    summary = commands.add_parser("model-summary")
    summary.add_argument("--config", required=True)
    summary.set_defaults(function=_model_summary)

    pruning = commands.add_parser("pruning-summary")
    pruning.add_argument("--source", required=True)
    pruning.add_argument("--target", required=True)
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
    args = build_parser().parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
