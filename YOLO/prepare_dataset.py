from pathlib import Path
import shutil
import sys


def read_data_yaml(yaml_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in yaml_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def copy_all_files(src_dir: Path, dst_dir: Path) -> int:
    copied = 0
    for src_file in src_dir.iterdir():
        if src_file.is_file():
            shutil.copy2(src_file, dst_dir / src_file.name)
            copied += 1
    return copied


def main() -> int:
    yolo_dir = Path(__file__).resolve().parent
    datasets_dir = yolo_dir / "datasets"
    train_images = datasets_dir / "train" / "images"
    train_labels = datasets_dir / "train" / "labels"
    val_images = datasets_dir / "val" / "images"
    val_labels = datasets_dir / "val" / "labels"
    data_yaml = yolo_dir / "data.yaml"

    if not train_images.exists() or not train_images.is_dir():
        raise FileNotFoundError(f"필수 경로 없음: {train_images}")
    if not train_labels.exists() or not train_labels.is_dir():
        raise FileNotFoundError(f"필수 경로 없음: {train_labels}")

    if not val_images.exists():
        val_images.mkdir(parents=True, exist_ok=True)
        print(f"[CREATE] {val_images}")
    else:
        print(f"[OK] 이미 존재: {val_images}")

    if not val_labels.exists():
        val_labels.mkdir(parents=True, exist_ok=True)
        print(f"[CREATE] {val_labels}")
    else:
        print(f"[OK] 이미 존재: {val_labels}")

    copied_images = copy_all_files(train_images, val_images)
    copied_labels = copy_all_files(train_labels, val_labels)
    print(f"[COPY] train/images -> val/images: {copied_images}개")
    print(f"[COPY] train/labels -> val/labels: {copied_labels}개")

    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml 없음: {data_yaml}")

    data = read_data_yaml(data_yaml)
    data_path = data.get("path", "")
    train_path = data.get("train", "")
    val_path = data.get("val", "")

    resolved_dataset_root = (data_yaml.parent / data_path).resolve()
    resolved_train_images = (resolved_dataset_root / train_path).resolve()
    resolved_val_images = (resolved_dataset_root / val_path).resolve()

    print(f"[CHECK] data.yaml path: {data_path} -> {resolved_dataset_root}")
    print(f"[CHECK] train 경로: {train_path} -> {resolved_train_images}")
    print(f"[CHECK] val 경로: {val_path} -> {resolved_val_images}")

    if resolved_train_images != train_images.resolve():
        print("[WARN] data.yaml train 경로가 현재 프로젝트 기준 train/images와 다릅니다.")
    else:
        print("[OK] data.yaml train 경로 정상")

    if resolved_val_images != val_images.resolve():
        print("[WARN] data.yaml val 경로가 현재 프로젝트 기준 val/images와 다릅니다.")
    else:
        print("[OK] data.yaml val 경로 정상")

    print("[DONE] 데이터셋 사전 점검 완료")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        raise SystemExit(1)
