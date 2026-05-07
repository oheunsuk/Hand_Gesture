from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
import random
import shutil
from typing import Any

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
DEFAULT_VAL_RATIO = 0.2
DEFAULT_TEST_RATIO = 0.2
DEFAULT_SEED = 42


@dataclass(frozen=True)
class Sample:
    class_name: str
    stem: str
    image_path: Path
    label_path: Path


@dataclass(frozen=True)
class SplitResult:
    total_samples: int
    train_samples: int
    val_samples: int
    test_samples: int
    class_distribution: dict[str, dict[str, int]]
    overlap_count: int
    dataset_root: str
    train_images: str
    val_images: str
    test_images: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_data_yaml(yaml_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in yaml_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def parse_class_name(stem: str) -> str:
    return stem.split("_", 1)[0] if "_" in stem else "unknown"


def find_image_path(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def collect_samples(images_dir: Path, labels_dir: Path) -> dict[str, Sample]:
    samples: dict[str, Sample] = {}
    if not labels_dir.exists():
        return samples
    for label_path in labels_dir.glob("*.txt"):
        stem = label_path.stem
        if stem in samples:
            continue
        image_path = find_image_path(images_dir, stem)
        if image_path is None:
            continue
        samples[stem] = Sample(
            class_name=parse_class_name(stem),
            stem=stem,
            image_path=image_path,
            label_path=label_path,
        )
    return samples


def clear_directory_files(target_dir: Path) -> None:
    if not target_dir.exists():
        return
    for file_path in target_dir.iterdir():
        if file_path.is_file():
            file_path.unlink()


def build_staging_samples(samples: list[Sample], staging_root: Path) -> list[Sample]:
    staging_images = staging_root / "images"
    staging_labels = staging_root / "labels"
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_images.mkdir(parents=True, exist_ok=True)
    staging_labels.mkdir(parents=True, exist_ok=True)

    staged_samples: list[Sample] = []
    for sample in samples:
        staged_image = staging_images / sample.image_path.name
        staged_label = staging_labels / sample.label_path.name
        shutil.copy2(sample.image_path, staged_image)
        shutil.copy2(sample.label_path, staged_label)
        staged_samples.append(
            Sample(
                class_name=sample.class_name,
                stem=sample.stem,
                image_path=staged_image,
                label_path=staged_label,
            )
        )
    return staged_samples


def split_samples_by_class(
    samples: list[Sample],
    val_ratio: float,
    test_ratio: float,
    rng: random.Random,
) -> tuple[list[Sample], list[Sample], list[Sample], dict[str, dict[str, int]]]:
    by_class: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_class[sample.class_name].append(sample)

    train_split: list[Sample] = []
    val_split: list[Sample] = []
    test_split: list[Sample] = []
    class_distribution: dict[str, dict[str, int]] = {}

    for class_name, class_samples in sorted(by_class.items()):
        class_samples = sorted(class_samples, key=lambda x: x.stem)
        rng.shuffle(class_samples)
        if test_ratio > 0 and len(class_samples) < 3:
            raise ValueError(f"클래스 '{class_name}' 샘플이 3개 미만이라 train/val/test 분할이 불가능합니다.")
        if test_ratio == 0 and len(class_samples) < 2:
            raise ValueError(f"클래스 '{class_name}' 샘플이 2개 미만입니다.")

        val_count = max(1, int(round(len(class_samples) * val_ratio))) if val_ratio > 0 else 0
        test_count = max(1, int(round(len(class_samples) * test_ratio))) if test_ratio > 0 else 0

        max_holdout = len(class_samples) - 1
        holdout_count = val_count + test_count
        if holdout_count > max_holdout:
            overflow = holdout_count - max_holdout
            reducible_val = max(0, val_count - (1 if val_ratio > 0 else 0))
            reduce_val = min(overflow, reducible_val)
            val_count -= reduce_val
            overflow -= reduce_val
            if overflow > 0:
                reducible_test = max(0, test_count - (1 if test_ratio > 0 else 0))
                reduce_test = min(overflow, reducible_test)
                test_count -= reduce_test
                overflow -= reduce_test
            if overflow > 0:
                raise ValueError(
                    f"클래스 '{class_name}' 샘플이 부족해 train/val/test 비율(val={val_ratio}, test={test_ratio}) 분할이 불가능합니다."
                )

        val_class = class_samples[:val_count]
        test_class = class_samples[val_count : val_count + test_count]
        train_class = class_samples[val_count + test_count :]
        train_split.extend(train_class)
        val_split.extend(val_class)
        test_split.extend(test_class)
        class_distribution[class_name] = {"train": len(train_class), "val": len(val_class), "test": len(test_class)}

    return train_split, val_split, test_split, class_distribution


def copy_split(samples: list[Sample], out_images: Path, out_labels: Path) -> None:
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        shutil.copy2(sample.image_path, out_images / sample.image_path.name)
        shutil.copy2(sample.label_path, out_labels / sample.label_path.name)


def validate_non_empty_class_count(samples: list[Sample], min_samples_per_class: int) -> None:
    class_counter = Counter(sample.class_name for sample in samples)
    if not class_counter:
        raise FileNotFoundError("분할할 샘플이 없습니다. data_collector.py로 데이터를 먼저 수집하세요.")
    for class_name, count in class_counter.items():
        if count < min_samples_per_class:
            raise ValueError(
                f"클래스 '{class_name}' 샘플이 {count}개입니다. 최소 {min_samples_per_class}개 이상 필요합니다."
            )


def prepare_dataset(
    yolo_dir: Path | None = None,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    seed: int = DEFAULT_SEED,
    min_samples_per_class: int = 2,
    verbose: bool = True,
) -> SplitResult:
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio는 0과 1 사이여야 합니다.")
    if not (0.0 <= test_ratio < 1.0):
        raise ValueError("test_ratio는 0 이상 1 미만이어야 합니다.")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio 합은 1보다 작아야 합니다.")

    base_dir = yolo_dir or Path(__file__).resolve().parent
    datasets_dir = base_dir / "datasets"
    train_images = datasets_dir / "train" / "images"
    train_labels = datasets_dir / "train" / "labels"
    val_images = datasets_dir / "val" / "images"
    val_labels = datasets_dir / "val" / "labels"
    test_images = datasets_dir / "test" / "images"
    test_labels = datasets_dir / "test" / "labels"
    data_yaml = base_dir / "data.yaml"

    if not train_images.exists() or not train_labels.exists():
        raise FileNotFoundError("datasets/train/images 또는 datasets/train/labels 경로가 없습니다.")
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml 없음: {data_yaml}")

    val_images.mkdir(parents=True, exist_ok=True)
    val_labels.mkdir(parents=True, exist_ok=True)
    test_images.mkdir(parents=True, exist_ok=True)
    test_labels.mkdir(parents=True, exist_ok=True)

    all_samples_map = collect_samples(train_images, train_labels)
    val_samples_map = collect_samples(val_images, val_labels)
    for stem, sample in val_samples_map.items():
        all_samples_map.setdefault(stem, sample)
    test_samples_map = collect_samples(test_images, test_labels)
    for stem, sample in test_samples_map.items():
        all_samples_map.setdefault(stem, sample)

    all_samples = list(all_samples_map.values())
    validate_non_empty_class_count(all_samples, min_samples_per_class=min_samples_per_class)

    rng = random.Random(seed)
    staging_root = datasets_dir / "_staging_split"
    staged_samples = build_staging_samples(all_samples, staging_root)
    train_split, val_split, test_split, class_distribution = split_samples_by_class(
        staged_samples,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        rng=rng,
    )

    clear_directory_files(train_images)
    clear_directory_files(train_labels)
    clear_directory_files(val_images)
    clear_directory_files(val_labels)
    clear_directory_files(test_images)
    clear_directory_files(test_labels)

    copy_split(train_split, train_images, train_labels)
    copy_split(val_split, val_images, val_labels)
    copy_split(test_split, test_images, test_labels)
    shutil.rmtree(staging_root, ignore_errors=True)

    train_names = {sample.stem for sample in train_split}
    val_names = {sample.stem for sample in val_split}
    test_names = {sample.stem for sample in test_split}
    overlap_count = len(train_names & val_names) + len(train_names & test_names) + len(val_names & test_names)
    if overlap_count > 0:
        raise RuntimeError("분할 후 train/val/test 파일명이 중복되었습니다. 분할 로직을 확인하세요.")

    data = read_data_yaml(data_yaml)
    dataset_root = (data_yaml.parent / data.get("path", "")).resolve()
    resolved_train_images = (dataset_root / data.get("train", "")).resolve()
    resolved_val_images = (dataset_root / data.get("val", "")).resolve()
    resolved_test_images = (dataset_root / data.get("test", "")).resolve()

    result = SplitResult(
        total_samples=len(all_samples),
        train_samples=len(train_split),
        val_samples=len(val_split),
        test_samples=len(test_split),
        class_distribution=class_distribution,
        overlap_count=overlap_count,
        dataset_root=str(dataset_root),
        train_images=str(resolved_train_images),
        val_images=str(resolved_val_images),
        test_images=str(resolved_test_images),
    )

    if verbose:
        print(
            f"[DATA] total={result.total_samples}, train={result.train_samples}, "
            f"val={result.val_samples}, test={result.test_samples}"
        )
        for class_name, stat in result.class_distribution.items():
            print(f"[SPLIT] {class_name}: train={stat['train']}, val={stat['val']}, test={stat['test']}")
        print(f"[CHECK] overlap={result.overlap_count}")
        print(f"[CHECK] train path={result.train_images}")
        print(f"[CHECK] val path={result.val_images}")
        print(f"[CHECK] test path={result.test_images}")

    return result


def main() -> int:
    try:
        prepare_dataset()
        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
