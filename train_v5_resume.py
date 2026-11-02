"""Colab에서 끊긴 v5(train_v5_aug, 7/30 에폭)를 로컬 GTX 1060에서 이어 학습
준비: Drive yolo_runs/train_v5_aug/weights/last.pt -> train_v5_aug/weights/last.pt
사용법: conda run --no-capture-output -n py311 python train_v5_resume.py
resume라 에폭·lr·증강(비/안개/어둡게/블러 등)은 체크포인트 설정 그대로 이어가고, 로컬에 맞는 것만 바꾼다"""
from pathlib import Path
import torch
from ultralytics import YOLO

HERE = Path(__file__).parent
RUN = HERE / "train_v5_aug"

def match_state_layout(trainer):
    """Colab에서 저장한 Adam 상태는 로컬 파라미터와 stride(메모리 배치)가 달라 fused AdamW가 거부함 -> 값은 그대로, 배치만 맞춤"""
    opt = trainer.optimizer
    for p, s in opt.state.items():
        for k in ("exp_avg", "exp_avg_sq"):
            if k in s and s[k].stride() != p.stride():
                s[k] = torch.empty_like(p).copy_(s[k])


if __name__ == "__main__":  # Windows에서 dataloader worker가 이 파일을 다시 import함
    model = YOLO(RUN / "weights" / "last.pt")
    model.add_callback("on_pretrain_routine_end", match_state_layout)  # resume_training(상태 로드) 직후 호출됨
    model.train(
        resume=True,
        data=str(HERE / "datasets" / "v4.yaml"),  # Colab 경로 대신 로컬 데이터
        save_dir=str(RUN),  # Drive 경로 대신 로컬, 기존 results.csv에 이어 씀
        batch=8,  # 3GB VRAM (Colab은 32, nbs=64라 기울기 누적으로 보정됨)
        cache=False,  # Colab은 disk 캐시였지만 로컬 디스크 여유가 부족
        workers=2,  # 0이면 비·안개 증강을 기다리느라 GPU가 절반쯤 놂, RAM 16GB라 2개까지만
    )
