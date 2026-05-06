import os
import random
import shutil
from pathlib import Path

def split_train_val(base_dir: Path, split_ratio: float = 0.2):
    # 1. 경로 설정
    train_img_dir = base_dir / "train" / "images"
    train_lbl_dir = base_dir / "train" / "labels"
    val_img_dir = base_dir / "val" / "images"
    val_lbl_dir = base_dir / "val" / "labels"

    # 2. 필수 경로 확인 및 생성
    for d in [val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if not train_img_dir.exists():
        print(f"[ERROR] 경로가 존재하지 않습니다: {train_img_dir}")
        return

    # 3. 이미지 파일 목록 가져오기
    images = [f for f in os.listdir(train_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not images:
        print("[ERROR] train/images 폴더가 비어 있습니다.")
        return

    # 4. 무작위 셔플 및 분할 개수 계산
    random.shuffle(images)
    val_count = int(len(images) * split_ratio)
    val_files = images[:val_count]

    print(f"[INFO] 총 데이터: {len(images)}개 -> Val 이동 목표: {val_count}개")

    # 5. 파일 이동 (Move)
    moved_count = 0
    for img_name in val_files:
        label_name = Path(img_name).stem + ".txt"
        
        src_img = train_img_dir / img_name
        dst_img = val_img_dir / img_name
        src_lbl = train_lbl_dir / label_name
        dst_lbl = val_lbl_dir / label_name

        # 이미지 이동
        if src_img.exists():
            shutil.move(str(src_img), str(dst_img))
            
            # 짝이 맞는 라벨 이동
            if src_lbl.exists():
                shutil.move(str(src_lbl), str(dst_lbl))
            moved_count += 1

    print(f"[DONE] {moved_count}개의 데이터셋을 val 폴더로 이동 완료했습니다.")

if __name__ == "__main__":
    # 현재 파일이 위치한 경로를 기준으로 datasets 폴더 탐색
    current_path = Path(__file__).resolve().parent / "datasets"
    split_train_val(current_path)