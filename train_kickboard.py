"""
COCO 사전학습 YOLO를 킥보드 데이터셋으로 파인튜닝

먼저 get_kickboard_data.py로 datasets/kickboard 를 받아둘 것

사용법:
    python train_kickboard.py                      # yolo11s, 50 epoch
    python train_kickboard.py --model yolo11m.pt --epochs 100
    작은 킥보드 대비 (v2에서 이어서, 평가는 make_small_test.py):
        python train_kickboard.py --model runs/kickboard/train_v2_e25/weights/best.pt --data datasets/kickboard_v2.yaml --scale 0.9 --epochs 15 --name train_v2_s09 --batch 8 --workers 0
    v3 (킥보드+사람+차+오토바이 한 모델, 먼저 make_v3_data.py):
        python train_kickboard.py --data datasets/v3.yaml --name train_v3 --batch 8 --workers 0
    학습 후 탐지: python detect.py --model runs/kickboard/train/weights/best.pt ... (TARGET_CLASSES를 {0: "kickboard"}로)
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).parent
DATA = HERE / "datasets" / "kickboard" / "data.yaml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo11s.pt", help="시작 가중치 (COCO 사전학습)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--fraction", type=float, default=1.0, help="학습에 쓸 데이터 비율 (빠른 테스트용)")
    parser.add_argument("--name", default="train", help="결과 폴더 이름 (runs/kickboard/<name>)")
    parser.add_argument("--data", default=str(DATA), help="데이터 yaml (v2: datasets/kickboard_v2.yaml)")
    parser.add_argument("--workers", type=int, default=4, help="데이터 로딩 프로세스 수 (윈도우에서 에러 나면 0)")
    parser.add_argument("--scale", type=float, default=0.5, help="스케일 증강 범위 (0.9면 최대 1/10까지 축소 → 멀리 있는 작은 킥보드 대비)")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        fraction=args.fraction,
        workers=args.workers,
        scale=args.scale,
        project=str((HERE / "runs" / "kickboard").resolve()),  # 상대경로면 runs/detect/runs/... 로 중첩 저장됨
        name=args.name,
        exist_ok=True,
    )

    # test 셋으로 최종 성능 확인
    metrics = model.val(data=args.data, split="test")
    print(f"test mAP50: {metrics.box.map50:.3f}  mAP50-95: {metrics.box.map:.3f}")
    print("최종 가중치:", HERE / "runs" / "kickboard" / args.name / "weights" / "best.pt")


if __name__ == "__main__":
    main()
