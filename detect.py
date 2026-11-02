"""
COCO 사전학습 YOLO로 사람 / 자전거 / 오토바이만 디텍션

사용법:
    python detect.py --source 이미지.jpg
    python detect.py --source 영상.mp4
    python detect.py --source 0          # 웹캠
    python detect.py --source test_images --no-show   # 폴더 일괄 처리
"""
import argparse
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

# COCO 클래스 id: 0=person, 1=bicycle, 3=motorcycle
TARGET_CLASSES = {0: "person", 1: "bicycle", 3: "motorcycle"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="이미지/영상 경로 또는 웹캠 번호")
    parser.add_argument("--model", default="yolo11n.pt", help="COCO로 학습된 YOLO 가중치")
    parser.add_argument("--conf", type=float, default=0.8)
    parser.add_argument("--no-show", action="store_true", help="창 없이 결과만 저장")
    args = parser.parse_args()

    model = YOLO(args.model)  # 없으면 자동 다운로드
    source = int(args.source) if args.source.isdigit() else args.source

    # stream=True: 영상/웹캠도 프레임 단위로 메모리 아끼며 처리
    results = model.predict(
        source=source,
        classes=list(TARGET_CLASSES),
        conf=args.conf,
        stream=True,
        save=True,
        project=str(Path("runs").resolve()),  # 상대경로면 runs/detect/runs/... 로 중첩 저장됨
        name="detect",
        exist_ok=True,
    )

    for r in results:
        counts = Counter(TARGET_CLASSES[int(c)] for c in r.boxes.cls)
        print(dict(counts))
        if args.no_show:
            continue

        frame = r.plot()
        y = 30
        for name in TARGET_CLASSES.values():
            cv2.putText(frame, f"{name}: {counts.get(name, 0)}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            y += 30

        cv2.imshow("YOLO COCO - person / bicycle / motorcycle", frame)
        # 이미지는 키 누를 때까지 대기, 영상은 q로 종료
        wait = 0 if r.path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")) else 1
        if cv2.waitKey(wait) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    print("결과 저장 위치: runs/detect")


if __name__ == "__main__":
    main()
