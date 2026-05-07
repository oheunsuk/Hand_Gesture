from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time
from typing import Any

import torch
from ultralytics import YOLO

from prepare_dataset import prepare_dataset, read_data_yaml

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy가 없는 환경 대응
    np = None


def parse_args() -> argparse.Namespace:
    yolo_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="YOLO 학습 파이프라인 (train/val/test 분할 + test 평가)")
    parser.add_argument("--data", type=str, default=str(yolo_dir / "data.yaml"))
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--project", type=str, default=str(yolo_dir / "runs"))
    parser.add_argument("--name", type=str, default="serbot_test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--min-samples-per-class", type=int, default=2)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if np is not None:
        np.random.seed(seed)


def resolve_device(device_arg: str, allow_cpu: bool) -> int | str:
    if device_arg.lower() == "cpu":
        if not allow_cpu:
            raise RuntimeError("CPU 학습은 비활성화되어 있습니다. --allow-cpu 옵션을 추가하세요.")
        return "cpu"
    if not torch.cuda.is_available():
        if allow_cpu:
            return "cpu"
        raise RuntimeError("CUDA를 찾지 못했습니다. GPU 드라이버/환경을 확인하거나 --allow-cpu를 사용하세요.")
    return int(device_arg)


def extract_metrics(metrics) -> dict[str, float]:
    box_metrics = metrics.box.mean_results()
    precision = float(box_metrics[0]) if len(box_metrics) > 0 else 0.0
    recall = float(box_metrics[1]) if len(box_metrics) > 1 else 0.0
    map50 = float(box_metrics[2]) if len(box_metrics) > 2 else 0.0
    map50_95 = float(box_metrics[3]) if len(box_metrics) > 3 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": map50,
        "map50_95": map50_95,
    }


def print_split_metrics(split_name: str, metric_dict: dict[str, float]) -> None:
    print(f"\n=== Metrics ({split_name} split) ===")
    print(f"Precision(정밀도): {metric_dict['precision']:.4f}")
    print(f"Recall(재현율):    {metric_dict['recall']:.4f}")
    print(f"F1 Score(F1 점수): {metric_dict['f1']:.4f}")
    print(f"mAP50(정확도 대체 지표): {metric_dict['map50']:.4f}")
    print(f"mAP50-95: {metric_dict['map50_95']:.4f}")


def ensure_output_dir(train_result, project_dir: Path, run_name: str) -> Path:
    save_dir = getattr(train_result, "save_dir", None)
    if save_dir is None:
        return project_dir / run_name
    return Path(save_dir)


def write_metrics_summary(
    save_dir: Path,
    args: argparse.Namespace,
    split_summary: dict[str, Any],
    val_metric_dict: dict[str, float],
    test_metric_dict: dict[str, float],
    elapsed_seconds: float,
) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / "metrics_summary.json"
    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "args": vars(args),
        "split_summary": split_summary,
        "val_metrics": val_metric_dict,
        "test_metrics": test_metric_dict,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def main() -> int:
    args = parse_args()
    set_global_seed(args.seed)
    yolo_dir = Path(__file__).resolve().parent
    data_yaml = Path(args.data).resolve()
    project_dir = Path(args.project).resolve()
    device = resolve_device(args.device, allow_cpu=args.allow_cpu)

    print(f"[INFO] 사용 장치: {device}")
    print(f"[INFO] train/val/test 분할 준비 시작 (val={args.val_ratio}, test={args.test_ratio})")

    split_result = prepare_dataset(
        yolo_dir=yolo_dir,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        min_samples_per_class=args.min_samples_per_class,
        verbose=True,
    )

    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml이 없습니다: {data_yaml}")
    data_yaml_values = read_data_yaml(data_yaml)
    if not data_yaml_values.get("test"):
        raise ValueError("data.yaml에 test 경로가 없습니다. 예: test: test/images")

    start_time = time.time()
    model = YOLO(args.model)
    amp_enabled = device != "cpu"
    train_result = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=device,
        batch=args.batch,
        amp=amp_enabled,
        project=str(project_dir),
        name=args.name,
        patience=args.patience,
        seed=args.seed,
    )

    val_metrics = model.val(data=str(data_yaml), split="val", device=device)
    val_metric_dict = extract_metrics(val_metrics)
    print_split_metrics("val", val_metric_dict)
    test_metrics = model.val(data=str(data_yaml), split="test", device=device)
    test_metric_dict = extract_metrics(test_metrics)
    print_split_metrics("test", test_metric_dict)

    save_dir = ensure_output_dir(train_result, project_dir=project_dir, run_name=args.name)
    output_path = write_metrics_summary(
        save_dir=save_dir,
        args=args,
        split_summary=split_result.to_dict(),
        val_metric_dict=val_metric_dict,
        test_metric_dict=test_metric_dict,
        elapsed_seconds=time.time() - start_time,
    )
    print(f"[INFO] metrics_summary 저장 완료: {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        raise SystemExit(1)