"""v3 평가: 클래스별 val mAP + 작은 킥보드 test(배율별)에서 s09와 비교
사용법: conda run -n py311 python tools/eval_v3.py <v3 best.pt> <s09 best.pt>"""
import sys
from pathlib import Path
from ultralytics import YOLO

HERE = Path(__file__).parent.parent
D = HERE / "datasets"
TMP = HERE / "runs" / "kickboard" / "eval_v3_tmp"
TMP.mkdir(parents=True, exist_ok=True)
v3, s09 = sys.argv[1], sys.argv[2]

# 1) 클래스별 val (v3 val = 킥보드·배경·야생동물·COCO 동물)
m = YOLO(v3)
r = m.val(data=str(D / "v3.yaml"), imgsz=640, batch=8, workers=0, plots=True, project=str(TMP), name="val_v3", exist_ok=True)
for i, c in enumerate(r.box.ap_class_index):
    print(f"[val] {m.names[int(c)]:11s} P {r.box.p[i]:.3f} R {r.box.r[i]:.3f} mAP50 {r.box.ap50[i]:.3f} mAP50-95 {r.box.ap[i]:.3f}", flush=True)

# 2) 킥보드 원본 test + 작은 킥보드 test, 클래스 이름은 모델 쪽에 맞춤 (킥보드 라벨은 0번이라 둘 다 같음)
sets = {"원본": (D / "kickboard", "test/images")} | {r_: (D / "kickboard_small_test", f"{r_}/images") for r_ in ["r0.5", "r0.25", "r0.125"]}
for w in (s09, v3):
    mm = YOLO(w)
    row = []
    for name, (root, sub) in sets.items():
        y = TMP / f"eval_{name}_{len(mm.names)}.yaml"
        y.write_text(f"path: {root.as_posix()}\ntrain: {sub}\nval: {sub}\ntest: {sub}\nnames: {list(mm.names.values())}\n", encoding="utf-8")
        rr = mm.val(data=str(y), split="test", imgsz=640, batch=8, classes=[0], plots=False, verbose=False, workers=0)
        row.append(f"{name} {rr.box.map50:.3f}")
    print(f"[kick] {Path(w).parts[-3]:14s}", " | ".join(row), flush=True)
