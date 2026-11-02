"""
학습된 킥보드 모델을 안드로이드용 TFLite(FP16/INT8)로 변환하고 PC에서 정확도·속도 비교

학습 환경(py311)에 tensorflow가 섞이지 않게 별도 환경(yolo_export, CPU torch, ultralytics==8.4.82)에서 실행:
    conda run -n yolo_export python export_mobile.py --model runs/kickboard/train/weights/best.pt
    conda run -n yolo_export python export_mobile.py --model runs/kickboard/train_v2_e25/weights/best.pt --imgsz 320
결과: 변환 로그 마지막에 출력되는 .tflite 경로
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).parent
DATA = HERE / "datasets" / "kickboard" / "data.yaml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="학습된 best.pt")
    parser.add_argument("--imgsz", type=int, default=640, help="모바일 입력 크기 (작을수록 빠름, 320/416/640)")
    parser.add_argument("--data", default=str(DATA), help="INT8 보정 + test 평가용 데이터 yaml")
    args = parser.parse_args()

    results = {}
    model = YOLO(args.model)
    results["pt"] = model.val(data=args.data, split="test", imgsz=args.imgsz, device="cpu", plots=False)

    # ultralytics 8.4.83+는 tflite→litert로 바뀌며 윈도우 변환 미지원 → yolo_export 환경은 8.4.82 고정
    for name, kw in {"fp16": {"half": True}, "int8": {"int8": True, "data": args.data}}.items():
        path = model.export(format="tflite", imgsz=args.imgsz, **kw)
        results[name] = YOLO(path, task="detect").val(data=args.data, split="test", imgsz=args.imgsz, device="cpu", plots=False)
        print(f"{name}: {path} ({Path(path).stat().st_size / 1e6:.1f} MB)")

    print(f"\n=== test 셋, imgsz {args.imgsz}, PC CPU 기준 (폰 속도와는 다름) ===")
    for name, m in results.items():
        print(f"{name:5s} mAP50 {m.box.map50:.3f}  mAP50-95 {m.box.map:.3f}  추론 {m.speed['inference']:.1f} ms/장")


if __name__ == "__main__":
    main()
