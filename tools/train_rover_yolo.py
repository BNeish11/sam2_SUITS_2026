from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO on the labeled rover dataset in the training folder."
    )
    parser.add_argument(
        "--dataset-yaml",
        type=Path,
        default=Path("training/rover images/rover_retrain_dataset/dataset.yaml"),
        help="Path to the YOLO dataset YAML.",
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        default=Path("yolo26n.pt"),
        help="Base YOLO weights for fine-tuning.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("runs/yolo_runs"),
        help="Ultralytics project output directory.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="rover_det_train",
        help="Ultralytics run name.",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.dataset_yaml.exists():
        raise FileNotFoundError(f"dataset-yaml not found: {args.dataset_yaml}")
    if not args.base_model.exists():
        raise FileNotFoundError(f"base-model not found: {args.base_model}")

    args.project.mkdir(parents=True, exist_ok=True)
    print(f"Training YOLO from: {args.base_model}")
    print(f"Dataset: {args.dataset_yaml}")

    model = YOLO(str(args.base_model))
    model.train(
        data=str(args.dataset_yaml.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(args.project.resolve()),
        name=args.run_name,
        exist_ok=True,
        seed=args.seed,
        workers=args.workers,
    )

    weights = args.project / args.run_name / "weights" / "best.pt"
    print(f"Done. best-weights: {weights}")


if __name__ == "__main__":
    main()