"""v4 평가: v3(대조군)와 v4를 같은 v4 val(오토바이 포함)에서 클래스별 비교 + 작은 킥보드 test(배율별)
사용법: conda run -n py311 python tools/eval_v4.py <v3 best.pt> <v4 best.pt>"""
import sys
from pathlib import Path
from ultralytics import YOLO

HERE = Path(__file__).parent.parent
D = HERE / "datasets"
TMP = HERE / "runs" / "kickboard" / "eval_v4_tmp"
TMP.mkdir(parents=True, exist_ok=True)
models = {"v3": sys.argv[1], "v4": sys.argv[2]}
sets = {"원본": (D / "kickboard", "test/images")} | {r_: (D / "kickboard_small_test", f"{r_}/images") for r_ in ["r0.5", "r0.25", "r0.125"]}

for tag, w in models.items():
    m = YOLO(w)
    r = m.val(data=str(D / "v4.yaml"), imgsz=640, batch=8, workers=0, plots=True, project=str(TMP), name=f"val_{tag}", exist_ok=True)
    for i, c in enumerate(r.box.ap_class_index):
        print(f"[val {tag}] {m.names[int(c)]:11s} P {r.box.p[i]:.3f} R {r.box.r[i]:.3f} mAP50 {r.box.ap50[i]:.3f} mAP50-95 {r.box.ap[i]:.3f}", flush=True)
    print(f"[val {tag}] all         mAP50 {r.box.map50:.3f} mAP50-95 {r.box.map:.3f}", flush=True)
    row = []
    for name, (root, sub) in sets.items():
        y = TMP / f"eval_{name}.yaml"
        y.write_text(f"path: {root.as_posix()}\ntrain: {sub}\nval: {sub}\ntest: {sub}\nnames: {list(m.names.values())}\n", encoding="utf-8")
        rr = m.val(data=str(y), split="test", imgsz=640, batch=8, classes=[0], plots=False, verbose=False, workers=0)
        row.append(f"{name} {rr.box.map50:.3f}")
    print(f"[kick {tag}]", " | ".join(row), flush=True)
