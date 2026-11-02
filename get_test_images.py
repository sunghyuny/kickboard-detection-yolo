"""
COCO val2017 어노테이션에서 사람 + 자전거/오토바이가 여러 개 있는 이미지만 골라 다운로드

사용법:
    python get_test_images.py --ann instances_val2017.json --num 50
"""
import argparse
import json
import random
import urllib.request
from collections import Counter
from pathlib import Path

PERSON, BICYCLE, MOTORCYCLE = 1, 2, 4  # COCO 원본 category_id (YOLO id와 다름)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann", required=True, help="instances_val2017.json 경로")
    parser.add_argument("--num", type=int, default=50)
    parser.add_argument("--min_objects", type=int, default=4, help="대상 객체 최소 개수")
    parser.add_argument("--out", default="test_images")
    args = parser.parse_args()

    data = json.load(open(args.ann, encoding="utf-8"))
    counts = {}
    for a in data["annotations"]:
        if a["category_id"] in (PERSON, BICYCLE, MOTORCYCLE) and not a["iscrowd"]:
            counts.setdefault(a["image_id"], Counter())[a["category_id"]] += 1

    # 사람 + 자전거 / 사람 + 오토바이 가 같이 있는 이미지를 반반씩
    def ok(c, vehicle):
        return c[PERSON] >= 1 and c[vehicle] >= 1 and sum(c.values()) >= args.min_objects

    random.seed(0)
    bikes = [i for i, c in counts.items() if ok(c, BICYCLE)]
    motos = [i for i, c in counts.items() if ok(c, MOTORCYCLE) and i not in bikes]
    half = args.num // 2
    picked = random.sample(bikes, min(half, len(bikes)))
    picked += random.sample(motos, min(args.num - len(picked), len(motos)))

    images = {img["id"]: img for img in data["images"]}
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    for n, img_id in enumerate(picked, 1):
        img = images[img_id]
        dst = out / img["file_name"]
        if not dst.exists():
            urllib.request.urlretrieve(img["coco_url"], dst)
        c = counts[img_id]
        print(f"[{n}/{len(picked)}] {img['file_name']}  "
              f"person={c[PERSON]} bicycle={c[BICYCLE]} motorcycle={c[MOTORCYCLE]}")


if __name__ == "__main__":
    main()
