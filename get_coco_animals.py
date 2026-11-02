"""
COCO에서 동물이 나오는 사진만 골라 받아 v3 클래스 YOLO 라벨로 저장

COCO 정답 라벨을 그대로 쓰므로 사람·차·오토바이도 의사 라벨이 아닌 정답이 같이 들어감
클래스: 0=kickboard(COCO엔 없음), 1=person, 2=car(버스·트럭 포함), 3=motorcycle, 4=animal(포유류 전부)

먼저 datasets/coco_src/annotations/instances_train2017.json 필요 (annotations_trainval2017.zip에서 추출)
사용법:
    python get_coco_animals.py                       # 고양이·개 각 1500장, 말·소·양·곰 각 300장
    python get_coco_animals.py --motorcycle 3000     # 동물 대신 오토바이 사진 → datasets/coco_motorcycle (v3 오토바이 부족 보충)
결과: datasets/coco_animals/{train,valid}/{images,labels}
"""
import argparse
import json
import random
import socket
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
ANN = HERE / "datasets" / "coco_src" / "annotations" / "instances_train2017.json"
OUT = HERE / "datasets" / "coco_animals"
URL = "http://images.cocodataset.org/train2017/"
# COCO category id(1부터, 중간 빈 번호 있음) → v3 클래스
COCO_TO_V3 = {1: 1, 3: 2, 6: 2, 8: 2, 4: 3} | {c: 4 for c in (17, 18, 19, 20, 21, 22, 23, 24, 25)}  # cat~giraffe
PICK = {17: "cat", 18: "dog", 19: "horse", 20: "sheep", 21: "cow", 23: "bear"}


def yolo_line(ann, w, h):
    x, y, bw, bh = ann["bbox"]
    return f"{COCO_TO_V3[ann['category_id']]} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pets", type=int, default=1500, help="고양이·개 각각 몇 장")
    parser.add_argument("--large", type=int, default=300, help="말·양·소·곰 각각 몇 장")
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--motorcycle", type=int, default=0, help="0이 아니면 동물 대신 오토바이 사진 이만큼")
    args = parser.parse_args()

    coco = json.loads(ANN.read_text())
    imgs = {i["id"]: i for i in coco["images"]}
    anns = defaultdict(list)
    for a in coco["annotations"]:
        if a["category_id"] in COCO_TO_V3 and not a["iscrowd"]:
            anns[a["image_id"]].append(a)

    out = HERE / "datasets" / "coco_motorcycle" if args.motorcycle else OUT
    pick = {4: "motorcycle"} if args.motorcycle else PICK
    rng = random.Random(0)
    chosen = set()
    taken = {int(f.stem) for f in OUT.rglob("*.txt")} if args.motorcycle else set()  # 동물 셋 사진과 겹치면 train/val이 섞임
    for cat, name in pick.items():
        ids = sorted({a["image_id"] for i, al in anns.items() for a in al if a["category_id"] == cat} - chosen - taken)
        n = args.motorcycle or (args.pets if name in ("cat", "dog") else args.large)
        chosen |= set(rng.sample(ids, min(n, len(ids))))
        print(f"{name}: 후보 {len(ids)}장 중 {min(n, len(ids))}장")
    chosen = sorted(chosen)
    rng.shuffle(chosen)
    n_val = int(len(chosen) * args.val)

    jobs = []
    for k, iid in enumerate(chosen):
        split = "valid" if k < n_val else "train"
        img = imgs[iid]
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)
        lines = [yolo_line(a, img["width"], img["height"]) for a in anns[iid]]
        (out / split / "labels" / f"{Path(img['file_name']).stem}.txt").write_text("\n".join(lines) + "\n")
        jobs.append((URL + img["file_name"], out / split / "images" / img["file_name"]))

    def fetch(job):
        url, path = job
        for k in range(5):  # COCO 서버가 동시 요청에 가끔 503을 줌
            if path.exists():
                return
            try:
                urllib.request.urlretrieve(url, path)
            except OSError:
                path.unlink(missing_ok=True)  # 반쯤 받은 파일이 남으면 다음 실행에서 건너뛰어 버림
                if k == 4:
                    raise
                time.sleep(2 ** k)

    socket.setdefaulttimeout(30)  # 응답 없는 요청 하나에 전체가 멈추지 않게 (멈추면 OSError → 재시도)
    with ThreadPoolExecutor(16) as pool:
        for k, _ in enumerate(pool.map(fetch, jobs), 1):
            if k % 500 == 0:
                print(f"받는 중 {k}/{len(jobs)}", flush=True)
    print(f"완료: train {len(chosen) - n_val}장, valid {n_val}장 → {out}")


if __name__ == "__main__":
    # 640x480 이미지의 고양이 박스 → animal(4), 정규화 중심 좌표
    assert yolo_line({"bbox": [0, 0, 320, 240], "category_id": 17}, 640, 480) == "4 0.250000 0.250000 0.500000 0.500000"
    main()
