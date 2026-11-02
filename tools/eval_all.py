"""v1~v5를 같은 시험셋으로 평가해 JSON으로 저장 (CPU, 학습 중인 GPU를 건드리지 않게)
- 킥보드 test 원본 + 작은 킥보드(r0.5/0.25/0.125): 모든 버전 공통 (킥보드 클래스만)
- v4 val 클래스별: 5클래스 모델(v3~)만
사용법: conda run --no-capture-output -n py311 python tools/eval_all.py <out.json> name=weights ..."""
import json
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

HERE = Path(__file__).parent.parent
D = HERE / "datasets"
TMP = HERE / "runs" / "kickboard" / "eval_all_tmp"
TMP.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)  # 학습 dataloader worker 몫 CPU를 남김
sets = {"원본": (D / "kickboard", "test/images")} | {r: (D / "kickboard_small_test", f"{r}/images") for r in ["r0.5", "r0.25", "r0.125"]}
kw = dict(imgsz=640, batch=8, device="cpu", workers=0, plots=False, verbose=False)

out = {}
for arg in sys.argv[2:]:
    tag, w = arg.split("=", 1)
    m = YOLO(w)
    res = {"kick": {}}
    for name, (root, sub) in sets.items():
        y = TMP / f"{tag}_{name}.yaml"
        y.write_text(f"path: {root.as_posix()}\ntrain: {sub}\nval: {sub}\ntest: {sub}\nnames: {list(m.names.values())}\n", encoding="utf-8")
        r = m.val(data=str(y), split="test", classes=[0], **kw)
        res["kick"][name] = [round(float(r.box.map50), 3), round(float(r.box.map), 3)]
    if len(m.names) == 5:
        r = m.val(data=str(D / "v4.yaml"), **kw)
        res["val"] = {m.names[int(c)]: [round(float(r.box.ap50[i]), 3), round(float(r.box.ap[i]), 3)] for i, c in enumerate(r.box.ap_class_index)}
        res["val"]["all"] = [round(float(r.box.map50), 3), round(float(r.box.map), 3)]
    out[tag] = res
    print(tag, json.dumps(res, ensure_ascii=False), flush=True)
    Path(sys.argv[1]).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
