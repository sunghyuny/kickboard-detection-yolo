"""
작은 킥보드 평가셋 만들기: kickboard test 이미지를 줄여서 회색 캔버스 가운데에 붙임

원래 test 셋은 킥보드가 크게 찍힌 사진뿐이라 mAP가 0.95로 포화됨 → 멀리 있는 킥보드 성능이 안 보임
배율별(1/2, 1/4, 1/8)로 폴더를 만들고 yaml 하나로 평가

사용법:
    python make_small_test.py
    yolo val model=runs/kickboard/train_v2_e25/weights/best.pt data=datasets/kickboard_small_test.yaml split=test
    (배율별로 보려면 --ratio 하나만 넣어 만들거나 아래 출력 yaml의 test 경로를 하나로 바꿔서 평가)
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
SRC = HERE / "datasets" / "kickboard" / "test"
OUT = HERE / "datasets" / "kickboard_small_test"
CANVAS = 640


def shrink_label(line, ratio, ox, oy, w, h):
    """원본 정규화 xywh → 캔버스 기준 정규화 xywh"""
    c, x, y, bw, bh = line.split()
    s = ratio * CANVAS / max(w, h)  # 원본 픽셀 → 캔버스 픽셀
    return f"{c} {(ox + float(x) * w * s) / CANVAS:.6f} {(oy + float(y) * h * s) / CANVAS:.6f} " \
           f"{float(bw) * w * s / CANVAS:.6f} {float(bh) * h * s / CANVAS:.6f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio", type=float, nargs="+", default=[0.5, 0.25, 0.125], help="캔버스 대비 이미지 긴 변 비율")
    args = parser.parse_args()

    dirs = []
    for ratio in args.ratio:
        name = f"r{ratio:g}"
        (OUT / name / "images").mkdir(parents=True, exist_ok=True)
        (OUT / name / "labels").mkdir(parents=True, exist_ok=True)
        for img_path in sorted((SRC / "images").iterdir()):
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            s = ratio * CANVAS / max(w, h)
            small = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
            canvas = np.full((CANVAS, CANVAS, 3), 114, np.uint8)
            ox, oy = (CANVAS - small.shape[1]) // 2, (CANVAS - small.shape[0]) // 2
            canvas[oy:oy + small.shape[0], ox:ox + small.shape[1]] = small
            cv2.imwrite(str(OUT / name / "images" / img_path.name), canvas)
            lines = (SRC / "labels" / f"{img_path.stem}.txt").read_text().split("\n")
            (OUT / name / "labels" / f"{img_path.stem}.txt").write_text(
                "\n".join(shrink_label(l, ratio, ox, oy, w, h) for l in lines if l.strip()) + "\n")
        dirs.append(f"{name}/images")
        print(f"{name}: 이미지 긴 변 {ratio * CANVAS:.0f}px")

    yaml = OUT.parent / "kickboard_small_test.yaml"
    # path를 안 쓰면 yaml이 있는 폴더 기준이라 PC 경로(사용자명)가 파일에 남지 않음
    dirs = [f"{OUT.name}/{d}" for d in dirs]
    yaml.write_text(f"# make_small_test.py로 생성. 배율: {args.ratio}\n"
                    f"train: {dirs[0]}\nval: {dirs[0]}\ntest:\n" + "".join(f"  - {d}\n" for d in dirs) +
                    "\nnc: 1\nnames: ['kickboard']\n", encoding="utf-8")
    print("완료 →", yaml)


if __name__ == "__main__":
    # 1/2 배율, 정사각 이미지: 가운데 박스는 그대로 가운데, 크기는 절반
    assert shrink_label("0 0.5 0.5 0.4 0.2", 0.5, 160, 160, 100, 100) == "0 0.500000 0.500000 0.200000 0.100000"
    main()
