"""
Roboflow Universe 야생동물(고라니·사슴·멧돼지·너구리 등) 데이터셋을 받아 animal 한 클래스로 합침

데이터셋마다 클래스 이름이 달라서 동물 이름만 animal로 남기고 나머지(사람·새 등)는 버림
사람·차는 make_v3_data.py가 COCO 모델 의사 라벨로 다시 붙임 (SPLITS 값 4 = 정답이 animal인 폴더)

API 키: 환경변수 ROBOFLOW_API_KEY 또는 홈 폴더의 .roboflow_key 파일
사용법:
    python get_wild_animals.py
결과: datasets/extra/<데이터셋>(원본), datasets/wild_animals/{train,valid}/{images,labels} (이미지는 하드링크)
"""
import os
import random
from pathlib import Path

import yaml
from roboflow import Roboflow

HERE = Path(__file__).parent
EXTRA = HERE / "datasets" / "extra"
OUT = HERE / "datasets" / "wild_animals"
# (workspace, project, version, 학습에 쓸 최대 장수)
SOURCES = [("roadanimals", "road-animals", 2, 800),
           ("seokjae", "waterdeer", 1, 400),
           ("drfghjkl", "wild_boar_detection", 1, 300)]
# 클래스 이름 → 동물이면 animal. waterdeer는 클래스 이름이 COCO 번호(15~23 = cat~giraffe)로 붙어 있음
ANIMALS = {"deer", "raccoon", "boar", "dog", "cat", "skunk", "opossum", "squirrel", "waterdeer", "wildboar",
           "wild boar", "feral hog"} | {str(c) for c in range(15, 24)}
VAL = 0.1


def to_animal(lines, names):
    """원본 YOLO 라벨에서 동물만 animal(4) 박스로 남김
    다각형(x1 y1 x2 y2 ...) 라벨은 박스로 바꿈: 한 파일에 박스와 다각형이 섞이면 ultralytics가 박스를 다각형으로 잘못 읽음"""
    out = []
    for l in lines:
        c, *v = l.split()
        if names[int(c)].lower() not in ANIMALS:
            continue
        v = list(map(float, v))
        if len(v) > 4:
            xs, ys = v[0::2], v[1::2]
            v = [(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)]
        out.append("4 " + " ".join(f"{x:.6f}" for x in v))
    return out


def main():
    key = os.environ.get("ROBOFLOW_API_KEY") or (Path.home() / ".roboflow_key").read_text().strip()
    rf = Roboflow(api_key=key)
    rng = random.Random(0)
    for ws, proj, ver, limit in SOURCES:
        src = EXTRA / proj
        if not (src / "data.yaml").exists():
            rf.workspace(ws).project(proj).version(ver).download("yolov11", location=str(src))
        names = yaml.safe_load((src / "data.yaml").read_text())["names"]
        names = list(names.values()) if isinstance(names, dict) else names
        # 원본 train/valid/test를 다 모아서 동물 박스가 있는 사진만 고름
        items = []
        for split in ("train", "valid", "test"):
            for lbl in sorted((src / split / "labels").glob("*.txt")) if (src / split).exists() else []:
                lines = to_animal([l for l in lbl.read_text().splitlines() if l.strip()], names)
                img = next((src / split / "images").glob(lbl.stem + ".*"), None)
                if lines and img:
                    items.append((img, lines))
        rng.shuffle(items)
        items = items[:limit]
        n_val = int(len(items) * VAL)
        for k, (img, lines) in enumerate(items):
            split = "valid" if k < n_val else "train"
            (OUT / split / "images").mkdir(parents=True, exist_ok=True)
            (OUT / split / "labels").mkdir(parents=True, exist_ok=True)
            dst = OUT / split / "images" / f"{proj}_{img.name}"
            if not dst.exists():
                os.link(img, dst)
            (OUT / split / "labels" / f"{proj}_{img.stem}.txt").write_text("\n".join(lines) + "\n")
        print(f"{proj}: 동물 있는 사진 중 {len(items)}장 사용 (valid {n_val}), 동물 박스 {sum(len(l) for _, l in items)}개")
    print("완료 →", OUT)


if __name__ == "__main__":
    # 사람(0)은 버리고 deer(1)만 animal(4)로
    assert to_animal(["0 0.1 0.1 0.1 0.1", "1 0.5 0.5 0.2 0.2"], ["person", "deer"]) == ["4 0.500000 0.500000 0.200000 0.200000"]
    # 다각형(사각형 꼭짓점 4개) → 박스
    assert to_animal(["0 0.4 0.4 0.6 0.4 0.6 0.6 0.4 0.6"], ["boar"]) == ["4 0.500000 0.500000 0.200000 0.200000"]
    main()
