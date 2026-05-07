import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camera_util import open_webcam


def parse_args() -> argparse.Namespace:
    yolo_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="YOLO 실시간 테스트 (정량 로그 포함)")
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="모델 경로(best.pt). 비워두면 runs에서 최신 best.pt 자동 선택",
    )
    parser.add_argument("--run-prefix", type=str, default="serbot_test", help="자동 검색할 run 접두사")
    parser.add_argument("--conf", type=float, default=0.5, help="confidence threshold")
    parser.add_argument("--interval", type=float, default=0.5, help="추론 간격(초)")
    parser.add_argument("--device", type=str, default="", help='추론 장치 (예: "cpu", "0")')
    parser.add_argument("--save-log", action="store_true", help="추론 로그를 jsonl로 저장")
    parser.add_argument(
        "--log-path",
        type=str,
        default=str(yolo_dir / "runs" / "yolo_test_metrics.jsonl"),
        help="로그 저장 경로(jsonl)",
    )
    return parser.parse_args()


def resolve_model_path(model_arg: str, run_prefix: str) -> Path:
    yolo_dir = Path(__file__).resolve().parent
    run_dir = yolo_dir / "runs"

    if model_arg:
        model_path = Path(model_arg).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"지정한 모델 파일이 없습니다: {model_path}")
        return model_path

    best_candidates = list(run_dir.glob(f"{run_prefix}*/weights/best.pt"))
    if best_candidates:
        return max(best_candidates, key=lambda p: p.stat().st_mtime)

    legacy_best = run_dir / run_prefix / "weights" / "best.pt"
    if legacy_best.exists():
        return legacy_best

    raise FileNotFoundError(f"best.pt 파일을 찾을 수 없습니다: {legacy_best}")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def get_label_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, list) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError(f"--conf 값은 0~1 사이여야 합니다: {args.conf}")
    if args.interval <= 0:
        raise ValueError(f"--interval 값은 0보다 커야 합니다: {args.interval}")

    model_path = resolve_model_path(args.model, args.run_prefix)
    log_path = Path(args.log_path).expanduser().resolve()

    print(f"[INFO] 사용 모델: {model_path}")
    print(f"[INFO] 설정: conf={args.conf}, interval={args.interval}s, device={args.device or 'auto'}")
    if args.save_log:
        print(f"[INFO] 로그 저장: {log_path}")

    model = YOLO(str(model_path))
    cap = None
    annotated_frame = None
    last_inference_time = 0.0
    frame_count = 0
    inference_count = 0
    total_inference_ms = 0.0
    loop_started_at = time.perf_counter()

    try:
        cap = open_webcam()
        print("테스트 시작 (q를 누르면 종료)...")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("[WARN] 카메라 프레임을 읽지 못해 종료합니다.")
                break

            frame_count += 1
            current_time = time.time()

            if current_time - last_inference_time >= args.interval:
                predict_kwargs: dict[str, Any] = {"conf": args.conf, "verbose": False}
                if args.device:
                    predict_kwargs["device"] = args.device

                inference_started_at = time.perf_counter()
                results = model(frame, **predict_kwargs)
                inference_ms = (time.perf_counter() - inference_started_at) * 1000.0

                result = results[0]
                annotated_frame = result.plot()
                last_inference_time = current_time
                inference_count += 1
                total_inference_ms += inference_ms

                label_counter: Counter[str] = Counter()
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        label_counter[get_label_name(model.names, class_id)] += 1

                detected_total = sum(label_counter.values())
                label_text = ", ".join(f"{label}:{count}" for label, count in sorted(label_counter.items()))
                if not label_text:
                    label_text = "없음"

                print(
                    f"[{time.strftime('%H:%M:%S')}] 감지={detected_total} | "
                    f"클래스={label_text} | 추론={inference_ms:.1f}ms"
                )

                if args.save_log:
                    append_jsonl(
                        log_path,
                        {
                            "type": "inference",
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "model": str(model_path),
                            "conf": args.conf,
                            "interval_sec": args.interval,
                            "device": args.device or "auto",
                            "inference_ms": round(inference_ms, 3),
                            "detected_total": detected_total,
                            "labels": dict(label_counter),
                        },
                    )

            display_frame = annotated_frame if annotated_frame is not None else frame
            cv2.imshow("YOLOv8 Interval Test", display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

    elapsed_sec = time.perf_counter() - loop_started_at
    avg_fps = (frame_count / elapsed_sec) if elapsed_sec > 0 else 0.0
    avg_inference_ms = (total_inference_ms / inference_count) if inference_count > 0 else 0.0

    summary = {
        "elapsed_sec": round(elapsed_sec, 3),
        "frame_count": frame_count,
        "inference_count": inference_count,
        "avg_fps": round(avg_fps, 3),
        "avg_inference_ms": round(avg_inference_ms, 3),
    }
    print(f"[INFO] 실행 요약: {summary}")

    if args.save_log:
        append_jsonl(
            log_path,
            {
                "type": "summary",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": str(model_path),
                "conf": args.conf,
                "interval_sec": args.interval,
                "device": args.device or "auto",
                **summary,
            },
        )
        print("[INFO] summary 로그 저장 완료")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"[ERROR] {exc}")
        raise SystemExit(1)