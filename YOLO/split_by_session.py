from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass(frozen=True)
class SessionSplit:
    train: list[str]
    val: list[str]
    test: list[str]


def parse_csv_names(value: str) -> list[str]:
    names = [name.strip() for name in value.split(",")]
    return [name for name in names if name]


def parse_args() -> argparse.Namespace:
    yolo_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="세션 단위 데이터셋을 train/val/test로 자동 분배합니다."
    )
    parser.add_argument(
        "--raw-root",
        type=str,
        default=str(yolo_dir / "datasets_raw"),
        help="세션 원본 루트 경로 (예: datasets_raw/session01/{images,labels})",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default=str(yolo_dir / "datasets"),
        help="YOLO 데이터셋 출력 루트 경로",
    )
    parser.add_argument("--train-sessions", type=str, required=True, help="train에 넣을 세션 목록(csv)")
    parser.add_argument("--val-sessions", type=str, required=True, help="val에 넣을 세션 목록(csv)")
    parser.add_argument("--test-sessions", type=str, required=True, help="test에 넣을 세션 목록(csv)")
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="출력 폴더 기존 파일 유지 (기본은 출력 폴더 파일 삭제 후 재분배)",
    )
    parser.add_argument("--dry-run", action="store_true", help="복사하지 않고 계획만 출력")
    return parser.parse_args()


def ensure_no_overlap(split: SessionSplit) -> None:
    train_set = set(split.train)
    val_set = set(split.val)
    test_set = set(split.test)
    overlap = (train_set & val_set) | (train_set & test_set) | (val_set & test_set)
    if overlap:
        overlap_text = ", ".join(sorted(overlap))
        raise ValueError(f"세션이 여러 split에 중복되었습니다: {overlap_text}")


def clear_split_files(split_root: Path) -> None:
    for sub in ("images", "labels"):
        target = split_root / sub
        target.mkdir(parents=True, exist_ok=True)
        for file_path in target.iterdir():
            if file_path.is_file():
                file_path.unlink()


def find_image_for_stem(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        image_path = images_dir / f"{stem}{ext}"
        if image_path.exists():
            return image_path
    return None


def copy_session_to_split(
    raw_root: Path,
    out_root: Path,
    split_name: str,
    session_name: str,
    dry_run: bool,
) -> tuple[int, int]:
    session_root = raw_root / session_name
    images_dir = session_root / "images"
    labels_dir = session_root / "labels"
    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"세션 경로가 잘못되었습니다: {session_root} (images/labels 필요)")

    out_images = out_root / split_name / "images"
    out_labels = out_root / split_name / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0
    for label_path in sorted(labels_dir.glob("*.txt")):
        stem = label_path.stem
        image_path = find_image_for_stem(images_dir, stem)
        if image_path is None:
            skipped += 1
            continue

        # 세션 prefix를 붙여 stem 충돌을 방지한다.
        target_stem = f"{session_name}_{stem}"
        target_label = out_labels / f"{target_stem}.txt"
        target_image = out_images / f"{target_stem}{image_path.suffix.lower()}"

        if not dry_run:
            shutil.copy2(label_path, target_label)
            shutil.copy2(image_path, target_image)
        copied += 1

    return copied, skipped


def copy_split_group(
    raw_root: Path,
    out_root: Path,
    split_name: str,
    sessions: list[str],
    dry_run: bool,
) -> tuple[int, int]:
    total_copied = 0
    total_skipped = 0
    for session_name in sessions:
        copied, skipped = copy_session_to_split(
            raw_root=raw_root,
            out_root=out_root,
            split_name=split_name,
            session_name=session_name,
            dry_run=dry_run,
        )
        total_copied += copied
        total_skipped += skipped
        print(f"[{split_name}] {session_name}: copied={copied}, skipped(no image)={skipped}")
    return total_copied, total_skipped


def main() -> int:
    args = parse_args()
    raw_root = Path(args.raw_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()

    split = SessionSplit(
        train=parse_csv_names(args.train_sessions),
        val=parse_csv_names(args.val_sessions),
        test=parse_csv_names(args.test_sessions),
    )
    if not split.train or not split.val or not split.test:
        raise ValueError("train/val/test 세션 목록은 모두 최소 1개 이상이어야 합니다.")
    ensure_no_overlap(split)

    if not raw_root.exists():
        raise FileNotFoundError(f"raw_root 경로가 없습니다: {raw_root}")

    if not args.keep_existing:
        for split_name in ("train", "val", "test"):
            clear_split_files(out_root / split_name)
        print("[INFO] 기존 datasets/train|val|test 파일을 정리했습니다.")

    train_copied, train_skipped = copy_split_group(raw_root, out_root, "train", split.train, args.dry_run)
    val_copied, val_skipped = copy_split_group(raw_root, out_root, "val", split.val, args.dry_run)
    test_copied, test_skipped = copy_split_group(raw_root, out_root, "test", split.test, args.dry_run)

    print(
        "[DONE] copied="
        f"train:{train_copied}, val:{val_copied}, test:{test_copied} | "
        f"skipped(no image)=train:{train_skipped}, val:{val_skipped}, test:{test_skipped}"
    )
    if args.dry_run:
        print("[INFO] dry-run 모드이므로 실제 파일 복사는 수행되지 않았습니다.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        raise SystemExit(1)
