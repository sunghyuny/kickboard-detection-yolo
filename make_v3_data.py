"""
v3 데이터셋 만들기: 킥보드 한 모델에 person / car / motorcycle 클래스를 합침

킥보드 데이터셋에는 사람·차 라벨이 없어서 그대로 합치면 "사람·차 = 배경"으로 학습됨
→ COCO 모델(yolo11m)로 같은 이미지에 사람·차·오토바이 의사 라벨(pseudo label)을 붙인다

클래스: 0=kickboard, 1=person, 2=car(버스·트럭 포함), 3=motorcycle, 4=animal(포유류 전부)
폴더마다 원래 정답 라벨이 무슨 클래스인지(SPLITS 값)로 바꿔 넣고, 나머지 클래스는 의사 라벨로 채움
COCO 동물 사진은 정답 라벨이 다 있어서 여기 대신 get_coco_animals.py로 만듦
이미지는 하드링크로 datasets/v3 아래에 걸어서 디스크를 더 쓰지 않음 (라벨만 새로 씀)

사용법:
    conda run -n py311 python make_v3_data.py               # 학습 중이라 GPU가 바쁘면 --device cpu
    conda run -n py311 python train_kickboard.py --data datasets/v3.yaml --name train_v3 --batch 8 --workers 0
"""
import argparse
import os
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).parent
SRC = HERE / "datasets"
OUT = SRC / "v3"
# 폴더 → 그 폴더 정답 라벨의 v3 클래스 (kickboard_v2.yaml에 쓰인 폴더 전부 + test)
SPLITS = {"kickboard/train": 0, "kickboard/valid": 0, "kickboard/test": 0,
          "extra/scooter-ewgvn/train": 0, "extra/escooter-detection/train": 0, "extra/escooter-detection/valid": 0,
          "backgrounds/train": 0, "backgrounds/valid": 0,
          "wild_animals/train": 4, "wild_animals/valid": 4}  # get_wild_animals.py 결과
# person, car/bus/truck → car, motorcycle, cat~giraffe → animal
COCO_TO_V3 = {0: 1, 2: 2, 5: 2, 7: 2, 3: 3} | {c: 4 for c in range(15, 24)}
KICK_IOU = 0.3  # 킥보드 정답 박스와 이만큼 겹치는 오토바이 의사 라벨은 킥보드를 잘못 본 것으로 보고 버림


def iou(a, b):
    """xywh(정규화) 박스 두 개의 IoU"""
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    inter = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    return inter / max(a[2] * a[3] + b[2] * b[3] - inter, 1e-9)


def merge(own_lines, pseudo, own):
    """폴더 정답 라벨(own 클래스로 바꿈) + 의사 라벨(cls, xywh)
    정답과 같은 클래스의 의사 라벨은 중복이라 버리고, 킥보드와 겹치는 오토바이도 버림"""
    out = [f"{own} " + l.split(maxsplit=1)[1] for l in own_lines]
    kicks = [list(map(float, l.split()[1:5])) for l in out] if own == 0 else []
    for c, box in pseudo:
        if c == own or (c == 3 and any(iou(box, k) > KICK_IOU for k in kicks)):
            continue
        out.append(f"{c} " + " ".join(f"{v:.6f}" for v in box))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", default=str(HERE / "yolo11m.pt"), help="의사 라벨용 COCO 모델 (클수록 라벨 품질↑)")
    parser.add_argument("--conf", type=float, default=0.35, help="의사 라벨 신뢰도 기준")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None, help="cpu로 주면 학습 중인 GPU를 건드리지 않음")
    args = parser.parse_args()

    teacher = YOLO(args.teacher)
    for split, own in SPLITS.items():
        src_img = SRC / split / "images"
        dst_img, dst_lbl = OUT / split / "images", OUT / split / "labels"
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)
        imgs = sorted(p for p in src_img.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
        counts = [0] * 5
        for i in range(0, len(imgs), args.batch):
            chunk = imgs[i:i + args.batch]
            results = teacher.predict([str(p) for p in chunk], conf=args.conf, classes=list(COCO_TO_V3),
                                      device=args.device, verbose=False)
            for img, r in zip(chunk, results):
                if not (dst_img / img.name).exists():
                    os.link(img, dst_img / img.name)
                own_txt = SRC / split / "labels" / f"{img.stem}.txt"  # 배경 폴더는 라벨 파일이 없음
                own_lines = [l for l in own_txt.read_text().splitlines() if l.strip()] if own_txt.exists() else []
                pseudo = [(COCO_TO_V3[int(c)], b.tolist()) for c, b in zip(r.boxes.cls, r.boxes.xywhn)]
                lines = merge(own_lines, pseudo, own)
                (dst_lbl / f"{img.stem}.txt").write_text("\n".join(lines) + "\n" if lines else "")
                for l in lines:
                    counts[int(l.split()[0])] += 1
        print(f"{split}: {len(imgs)}장  kickboard {counts[0]}  person {counts[1]}  car {counts[2]}  motorcycle {counts[3]}  animal {counts[4]}")
    print("완료 →", OUT, "/ 학습 yaml: datasets/v3.yaml")


if __name__ == "__main__":
    # 오토바이 의사 라벨이 킥보드 정답과 겹치면 빠지는지 확인
    k = ["0 0.5 0.5 0.2 0.4"]
    assert merge(k, [(3, [0.5, 0.5, 0.2, 0.4]), (1, [0.5, 0.5, 0.2, 0.4]), (3, [0.1, 0.1, 0.1, 0.1])], 0) == \
        k + ["1 0.500000 0.500000 0.200000 0.400000", "3 0.100000 0.100000 0.100000 0.100000"]
    # 동물 데이터셋: 원래 클래스(예: deer=2)는 animal(4)로, 같은 동물 의사 라벨은 중복이라 버림
    assert merge(["2 0.5 0.5 0.2 0.4"], [(4, [0.5, 0.5, 0.2, 0.4]), (1, [0.1, 0.1, 0.1, 0.1])], 4) == \
        ["4 0.5 0.5 0.2 0.4", "1 0.100000 0.100000 0.100000 0.100000"]
    main()
