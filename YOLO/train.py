from __future__ import annotations

import argparse
import csv
from collections import Counter
import json
from pathlib import Path
import random
import time
from typing import Any

import torch
from ultralytics import YOLO

from prepare_dataset import collect_samples, read_data_yaml

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy가 없는 환경 대응
    np = None


def parse_args() -> argparse.Namespace:
    yolo_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="YOLO 학습 파이프라인 (고정 train/val/test 사용 + test 평가)")
    parser.add_argument("--data", type=str, default=str(yolo_dir / "data.yaml"))
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--project", type=str, default=str(yolo_dir / "runs"))
    parser.add_argument("--name", type=str, default="serbot_test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=20)
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


def summarize_fixed_splits(data_yaml: Path, data_yaml_values: dict[str, str]) -> dict[str, Any]:
    dataset_root = (data_yaml.parent / data_yaml_values.get("path", "")).resolve()
    split_samples: dict[str, dict[str, Any]] = {}

    for split_name in ("train", "val", "test"):
        images_rel = data_yaml_values.get(split_name)
        if not images_rel:
            raise ValueError(f"data.yaml에 {split_name} 경로가 없습니다. 예: {split_name}: {split_name}/images")

        images_dir = (dataset_root / images_rel).resolve()
        labels_dir = images_dir.parent / "labels"
        if not images_dir.exists() or not labels_dir.exists():
            raise FileNotFoundError(
                f"{split_name} 경로가 올바르지 않습니다. images={images_dir}, labels={labels_dir}"
            )

        sample_map = collect_samples(images_dir=images_dir, labels_dir=labels_dir)
        split_samples[split_name] = {
            "samples": list(sample_map.values()),
            "images_dir": str(images_dir),
        }

    train_stems = {sample.stem for sample in split_samples["train"]["samples"]}
    val_stems = {sample.stem for sample in split_samples["val"]["samples"]}
    test_stems = {sample.stem for sample in split_samples["test"]["samples"]}
    overlap_count = len(train_stems & val_stems) + len(train_stems & test_stems) + len(val_stems & test_stems)
    if overlap_count > 0:
        raise RuntimeError("고정 split에서 파일명이 중복되었습니다. train/val/test 구성을 확인하세요.")

    class_distribution: dict[str, dict[str, int]] = {}
    class_names = {
        sample.class_name for split_name in ("train", "val", "test") for sample in split_samples[split_name]["samples"]
    }
    for class_name in sorted(class_names):
        class_distribution[class_name] = {}
        for split_name in ("train", "val", "test"):
            class_counter = Counter(sample.class_name for sample in split_samples[split_name]["samples"])
            class_distribution[class_name][split_name] = class_counter.get(class_name, 0)

    train_count = len(split_samples["train"]["samples"])
    val_count = len(split_samples["val"]["samples"])
    test_count = len(split_samples["test"]["samples"])
    total_samples = train_count + val_count + test_count
    if total_samples == 0:
        raise FileNotFoundError("학습할 샘플이 없습니다. datasets/train|val|test 데이터를 확인하세요.")

    return {
        "total_samples": total_samples,
        "train_samples": train_count,
        "val_samples": val_count,
        "test_samples": test_count,
        "class_distribution": class_distribution,
        "overlap_count": overlap_count,
        "dataset_root": str(dataset_root),
        "train_images": split_samples["train"]["images_dir"],
        "val_images": split_samples["val"]["images_dir"],
        "test_images": split_samples["test"]["images_dir"],
    }


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


def save_compact_training_figure(save_dir: Path) -> Path | None:
    results_csv = save_dir / "results.csv"
    if not results_csv.exists():
        print(f"[WARN] results.csv를 찾지 못했습니다: {results_csv}")
        return None

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib이 없어 custom 4패널 그래프 생성을 건너뜁니다.")
        return None

    epochs: list[int] = []
    train_box_loss: list[float] = []
    val_box_loss: list[float] = []
    map50_values: list[float] = []
    precision_values: list[float] = []
    recall_values: list[float] = []

    with results_csv.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            epoch_raw = row.get("epoch")
            train_loss_raw = row.get("train/box_loss")
            val_loss_raw = row.get("val/box_loss")
            map50_raw = row.get("metrics/mAP50(B)")
            precision_raw = row.get("metrics/precision(B)")
            recall_raw = row.get("metrics/recall(B)")

            if not all([epoch_raw, train_loss_raw, val_loss_raw, map50_raw, precision_raw, recall_raw]):
                continue

            try:
                epochs.append(int(float(epoch_raw)))
                train_box_loss.append(float(train_loss_raw))
                val_box_loss.append(float(val_loss_raw))
                map50_values.append(float(map50_raw))
                precision_values.append(float(precision_raw))
                recall_values.append(float(recall_raw))
            except ValueError:
                continue

    if not epochs:
        print(f"[WARN] results.csv에서 그래프용 데이터를 읽지 못했습니다: {results_csv}")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax_loss, ax_map50, ax_precision, ax_recall = axes.ravel()

    ax_loss.plot(epochs, train_box_loss, label="train/box_loss")
    ax_loss.plot(epochs, val_box_loss, label="val/box_loss")
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend()

    ax_map50.plot(epochs, map50_values, color="tab:green")
    ax_map50.set_title("mAP50")
    ax_map50.set_xlabel("Epoch")
    ax_map50.grid(True, alpha=0.3)

    ax_precision.plot(epochs, precision_values, color="tab:orange")
    ax_precision.set_title("Precision")
    ax_precision.set_xlabel("Epoch")
    ax_precision.grid(True, alpha=0.3)

    ax_recall.plot(epochs, recall_values, color="tab:red")
    ax_recall.set_title("Recall")
    ax_recall.set_xlabel("Epoch")
    ax_recall.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path = save_dir / "custom_metrics_4panel.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main() -> int:
    args = parse_args()
    set_global_seed(args.seed)
    yolo_dir = Path(__file__).resolve().parent
    data_yaml = Path(args.data).resolve()
    project_dir = Path(args.project).resolve()
    device = resolve_device(args.device, allow_cpu=args.allow_cpu)

    print(f"[INFO] 사용 장치: {device}")
    print("[INFO] 고정 split(train/val/test) 검증 시작")

    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml이 없습니다: {data_yaml}")
    data_yaml_values = read_data_yaml(data_yaml)
    split_summary = summarize_fixed_splits(data_yaml, data_yaml_values)
    print(
        f"[DATA] total={split_summary['total_samples']}, train={split_summary['train_samples']}, "
        f"val={split_summary['val_samples']}, test={split_summary['test_samples']}"
    )
    print(f"[CHECK] overlap={split_summary['overlap_count']}")
    print(f"[CHECK] train path={split_summary['train_images']}")
    print(f"[CHECK] val path={split_summary['val_images']}")
    print(f"[CHECK] test path={split_summary['test_images']}")

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
        split_summary=split_summary,
        val_metric_dict=val_metric_dict,
        test_metric_dict=test_metric_dict,
        elapsed_seconds=time.time() - start_time,
    )
    print(f"[INFO] metrics_summary 저장 완료: {output_path}")
    plot_path = save_compact_training_figure(save_dir)
    if plot_path is not None:
        print(f"[INFO] 4패널 그래프 저장 완료: {plot_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        raise SystemExit(1)