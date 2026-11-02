"""
YOLOE 제로샷(학습 없이 텍스트로 클래스 지정)으로 응급차량 + 사람/자전거/오토바이 디텍션 테스트

test_emergency/<클래스>/ 폴더의 각 이미지에서 해당 클래스가 1개 이상 잡혔는지(이미지 단위 recall) 확인

사용법:
    python test_yoloe.py --model yoloe-11s-seg.pt --conf 0.25
"""
import argparse
from collections import Counter
from pathlib import Path

from ultralytics import YOLOE

# 앞 4개는 COCO id 순서(0 person, 1 bicycle, 3 motorcycle)와 맞춰서 COCO 평가 코드 재사용 가능하게
CLASSES = ["person", "bicycle", "car", "motorcycle",
           "police car", "ambulance", "fire truck", "truck", "bus"]
TARGETS = {"person", "bicycle", "motorcycle", "police car", "ambulance", "fire truck"}
FOLDER_TO_CLASS = {"police_car": "police car", "ambulance": "ambulance", "fire_truck": "fire truck"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yoloe-11s-seg.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--data", default="test_emergency")
    args = parser.parse_args()

    model = YOLOE(args.model)
    # car/truck/bus는 일반 차량이 응급차로 잘못 분류되지 않게 넣는 '비교용' 클래스 (결과에선 숨김)
    model.set_classes(CLASSES)
    keep = [i for i, n in enumerate(CLASSES) if n in TARGETS]

    out = Path("runs/yoloe") / Path(args.model).stem
    for folder, target in FOLDER_TO_CLASS.items():
        imgs = sorted((Path(args.data) / folder).glob("*.jpg"))
        hit, confused = 0, Counter()
        for p in imgs:
            r = model.predict(p, conf=args.conf, classes=keep, verbose=False)[0]
            found = Counter(CLASSES[int(c)] for c in r.boxes.cls)
            if found[target]:
                hit += 1
            for name in FOLDER_TO_CLASS.values():
                if name != target and found[name]:
                    confused[name] += 1
            r.save(filename=str(out / folder / p.name))
        print(f"{target:<11} {hit}/{len(imgs)} 이미지에서 검출 (recall {hit/len(imgs):.2f})"
              f"  다른 응급차로 오검출: {dict(confused)}")
    print("결과 이미지:", out)


if __name__ == "__main__":
    main()
