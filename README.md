# 킥보드·사람·차·오토바이·동물 탐지 (YOLO11s)

폰 카메라로 도로 위 **전동킥보드, 사람, 차, 오토바이, 동물**을 실시간 탐지하는 모바일 앱용 모델입니다.
COCO에는 킥보드 클래스가 없어서, COCO로 사전학습된 YOLO11s를 킥보드 데이터로 파인튜닝한 뒤 다섯 클래스를 한 모델에 합쳤습니다.
모든 버전이 같은 구조(YOLO11s, 9.4M 파라미터, 640px)여서 폰에서 드는 추론 비용은 버전 간에 같습니다.

## 버전별 비교

자세한 표: [report/report_v1_v5.pdf](report/report_v1_v5.pdf) (v5 시험 포함) · [report/report_v1_v4.pdf](report/report_v1_v4.pdf)

| 버전 | 보완한 점 | 킥보드 test mAP50<br>(원본 / 1/8 축소) | v4 검증셋 전체 mAP50 |
|---|---|---|---|
| v1 | 킥보드 전용 모델 첫 학습 | 0.951 / 0.392 | — |
| v2 | Roboflow 킥보드 데이터 추가, 한국 거리 배경 사진으로 오검출 감소, 작은 킥보드 대응 학습(scale 0.9) | 0.962 / 0.745 | — |
| v3 | 5클래스 통합 (폰에서 모델 1개만 실행), 동물 클래스 추가 | 0.963 / 0.820 | 0.617 |
| **v4** | COCO 오토바이 사진 보충 (오토바이 mAP50 0.335 → 0.682) | **0.977** / 0.814 | **0.723** |
| v5 | v4에서 이어 30 epoch, 학습 이미지 절반에 비·안개·밤·모션 블러·초점 흐림 합성 증강 (흐린 장면 재현율 향상) | 미측정 | 0.737\* |

- 킥보드 test: 학습에 쓰지 않은 68장. "1/8 축소"는 이미지를 1/8로 줄여 멀리 있는 작은 킥보드를 흉내 낸 것입니다.
- v1·v2는 킥보드만 탐지하는 모델이라 5클래스 검증셋 점수가 없습니다.
- \* v5는 같은 v4 검증셋이지만 `eval_all.py` 재측정이 아니라 학습 중 마지막 에폭 기록입니다(v4는 같은 방식으로 0.724). 킥보드 test는 아직 재지 않았습니다.

### v5: 흐린 장면에서 놓치는 비율

v5의 목표는 빠르게 움직이거나 비·밤처럼 흐린 장면에서 덜 놓치는 것이라, 선명한 검증셋 점수보다 흐린 이미지의 재현율로 비교했습니다.
COCO val2017 600장(v4·v5 모두 학습에 쓰지 않은 분할)에 가로 모션 블러를 입히고, 정답 박스와 IoU 0.5 이상으로 맞는 탐지가 신뢰도 0.4 이상인 비율입니다.

| 클래스 (정답 박스 수) | 블러 없음 v4 → v5 | 15px 블러 v4 → v5 | 25px 블러 v4 → v5 |
|---|---|---|---|
| 사람 (2,070) | 0.408 → 0.414 | 0.151 → **0.192** | 0.066 → 0.075 |
| 차 (889) | 0.476 → 0.478 | 0.190 → **0.273** | 0.111 → 0.155 |
| 오토바이 (346) | 0.497 → 0.488 | 0.165 → **0.277** | 0.075 → 0.142 |
| 동물 (465) | 0.576 → 0.568 | 0.217 → **0.331** | 0.153 → 0.185 |
| 오탐 (100장당, 신뢰도 0.4 이상) | 53.3 → 49.2 | 26.5 → 23.8 | 22.2 → 16.3 |

- 블러가 없을 때는 거의 같고, v5가 학습한 블러 범위(7–15px)인 15px에서 재현율이 약 1.3–1.7배로 올랐습니다. 학습 범위 밖인 25px에서는 개선 폭이 작습니다.
- COCO 사진은 차량 전방 시점이 아니고 블러도 합성이라, 실제 주행 영상과는 차이가 있을 수 있습니다.
- v5는 v4보다 30 epoch을 더 학습했기 때문에, 증강 없이 같은 epoch을 더 학습한 모델과 비교하지 않는 한 개선이 모두 증강 덕이라고 단정할 수는 없습니다.

## 가중치

`train_v4_moto/weights/best.pt` (v4, 클래스: `kickboard, person, car, motorcycle, animal`)

v5 가중치(`train_v5_aug`)는 아직 저장소에 올리지 않았습니다. 클래스와 사용법은 v4와 같습니다.

```python
from ultralytics import YOLO
model = YOLO("train_v4_moto/weights/best.pt")
model.predict("image.jpg", conf=0.4, save=True)
```

영상 탐지(ByteTrack 추적 + 박스 평활화):

```
python detect_video.py --source 영상.mp4 --coco "" --kick train_v4_moto/weights/best.pt
```

## 파일 구성

| 파일 | 내용 |
|---|---|
| `get_kickboard_data.py`, `get_wild_animals.py`, `get_coco_animals.py` | 데이터 다운로드 (Roboflow API 키는 환경변수 `ROBOFLOW_API_KEY`) |
| `make_v3_data.py` | 킥보드 데이터에 사람·차·오토바이 의사 라벨(yolo11m)을 붙여 5클래스 데이터셋 생성 |
| `make_small_test.py` | 작은 킥보드 test 세트(이미지 축소) 생성 |
| `train_kickboard.py` | 로컬 학습 (v1·v2) |
| `colab_train_v3.ipynb`, `colab_train_v4.ipynb` | Colab T4 학습 (v3·v4) |
| `train_v5_resume.py` | 끊긴 Colab 학습을 로컬에서 이어 학습 |
| `detect.py`, `detect_video.py` | 사진·영상 탐지 |
| `export_mobile.py` | 안드로이드용 TFLite(FP16/INT8) 변환 |
| `tools/eval_all.py`, `tools/make_report.py` | 전 버전 같은 시험셋 평가 → 비교 리포트 |
| `datasets/*.yaml` | 데이터셋 구성 (이미지는 저장소에 없음, 위 스크립트로 받음) |

## 데이터 출처

데이터셋 이미지는 저장소에 포함하지 않았습니다.

- 킥보드: Roboflow Universe [kickboard-dfkjo](https://universe.roboflow.com/dklyolo1/kickboard-dfkjo), scooter-ewgvn, escooter-detection
- 야생동물: Roboflow Universe road-animals, waterdeer, wild_boar_detection
- 사람·차·오토바이·동물 사진: [COCO 2017](https://cocodataset.org/)

각 데이터셋의 라이선스는 원본 페이지를 따릅니다.

## 라이선스

[Ultralytics](https://github.com/ultralytics/ultralytics)(AGPL-3.0)로 학습한 모델이라, 가중치도 AGPL-3.0 조건을 따릅니다.
