"""eval_all.json -> v1~v5 비교표 HTML (PDF는 Chrome headless로 인쇄)
사용법: conda run -n py311 python tools/make_report.py runs/kickboard/eval_all.json runs/kickboard/report_v1_v5.html"""
import json
import sys
from pathlib import Path

ev = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
V5 = next((k for k in ev if k.startswith("v5")), None)  # 없으면 v1~v4만

# 버전별: 이름, 가중치 폴더, 출발점, 클래스, 학습, 데이터, 보완한 점
VERS = [
    ("v1", "runs/kickboard/train", "yolo11s (COCO)", "1 (킥보드)", "20 epoch · 로컬",
     "킥보드 데이터 1,392장",
     "첫 버전. COCO에는 킥보드 클래스가 없어서 킥보드 전용 모델을 직접 학습. 차·오토바이·사람은 COCO 모델을 따로 돌림"),
    ("v2", "runs/kickboard/train_v2_s09", "yolo11s → v2 25ep", "1 (킥보드)", "25 + 15 epoch · 로컬",
     "v1 + Roboflow 킥보드 2종(약 7,300장) + 한국 거리 배경(라벨 없음)",
     "킥보드 데이터 6배로 늘림. 킥보드 없는 한국 거리 사진을 넣어 도로·간판을 킥보드로 잡는 오검출 감소. scale 0.9로 추가 학습해 멀리 있는 작은 킥보드 대응"),
    ("v3", "train_v3_5cls", "v2", "5", "30 epoch · Colab",
     "v2 + 사람·차·오토바이 의사 라벨(yolo11m) + 야생동물(고라니·멧돼지 등) + COCO 동물 사진",
     "한 모델로 킥보드·사람·차·오토바이·동물을 모두 탐지 → 폰에서 모델 2개 돌리던 추론 비용 제거. 동물 클래스 추가"),
    ("v4", "train_v4_moto", "v2 (v3와 같은 출발점)", "5", "30 epoch · Colab",
     "v3 + COCO 오토바이 사진 2,700장(정답 라벨)",
     "v3 오토바이 성능 부족(재현율 0.25) 보완. 학습 데이터만 바꾼 대조 실험"),
    ("v5", "train_v5_aug", "v4", "5", "30 epoch · Colab 8 + 로컬",
     "v4와 같은 데이터",
     "비(폭우·이슬비)·안개·밤(어둡게+노이즈)·모션블러·초점 흐림·JPEG 압축 합성 증강 → 악천후·저화질 카메라 대응"),
]
KICK = ["원본", "r0.5", "r0.25", "r0.125"]
CLS = ["kickboard", "person", "car", "motorcycle", "animal", "all"]
CLS_KO = ["킥보드", "사람", "차", "오토바이", "동물", "전체"]


def cells(vals, best):
    return "".join(f'<td class="num{" best" if v is not None and v == best[i] else ""}">{"—" if v is None else f"{v:.3f}"}</td>' for i, v in enumerate(vals))


def col_best(rows):
    return [max((r[i] for r in rows if r[i] is not None), default=None) for i in range(len(rows[0]))]


tags = ["v1", "v2", "v3", "v4"] + ([V5] if V5 else [])
kick = {t: [ev[t]["kick"][k][0] for k in KICK] for t in tags}
val = {t: [ev[t]["val"][c][0] if "val" in ev[t] else None for c in CLS] for t in tags}
kb, vb = col_best(list(kick.values())), col_best(list(val.values()))
v5_label = "v5" if V5 in (None, "v5") else f"v5<br><small>학습 중 ({V5.split('ep')[-1]}/30)</small>"
VERS = [v for v in VERS if v[0] != "v5" or V5]

rows_info = "".join(
    f"<tr><th>{v5_label if n == 'v5' else n}</th><td>{start}</td><td class='c'>{cls}</td><td>{train}</td><td>{data}</td><td class='why'>{why}</td></tr>"
    for n, _, start, cls, train, data, why in VERS)
rows_res = "".join(
    f"<tr><th>{v5_label if t == V5 else t}</th>{cells(kick[t], kb)}{cells(val[t], vb)}</tr>" for t in tags)

V5_NOTE = ("<li><b>v5는 학습 중간 가중치</b>라 최종 성능이 아님. 날씨 증강 효과는 맑은 날 시험셋보다 악천후 영상에서 확인해야 함.</li>" if V5 and V5 != "v5" else "")
html = f"""<!doctype html><html lang="ko"><meta charset="utf-8"><title>킥보드 탐지 모델 비교</title>
<style>
@page {{ size: A4 landscape; margin: 12mm; }}
body {{ font-family: 'Malgun Gothic', sans-serif; font-size: 10.5px; color: #1d2433; }}
h1 {{ font-size: 18px; margin: 0 0 2px; }} h2 {{ font-size: 13px; margin: 14px 0 5px; }}
.sub {{ color: #5b6475; margin-bottom: 8px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #c9cfda; padding: 4px 6px; vertical-align: top; }}
thead th {{ background: #eef1f6; text-align: center; vertical-align: middle; }}
tbody th {{ background: #f7f8fb; width: 60px; text-align: center; vertical-align: middle; }}
.num {{ text-align: right; font-variant-numeric: tabular-nums; }} .c {{ text-align: center; }}
.best {{ font-weight: bold; background: #e3f2e6; }}
.why {{ width: 34%; }}
ul {{ margin: 4px 0; padding-left: 16px; }} li {{ margin: 2px 0; }}
small {{ color: #8a5a00; font-weight: normal; }}
</style>
<h1>킥보드 탐지 모델 v1 ~ {tags[-1][:2]} 비교</h1>
<div class="sub">YOLO11s (9.4M 파라미터, 640px) · 모든 버전 같은 구조라 폰 추론 비용 동일 · 2026-11-02 기준, 같은 시험셋으로 다시 측정</div>

<h2>1. 버전별 보완 내용</h2>
<table><thead><tr><th>버전</th><th>출발 가중치</th><th>클래스</th><th>학습</th><th>학습 데이터</th><th>보완한 점</th></tr></thead>
<tbody>{rows_info}</tbody></table>

<h2>2. 성능 (mAP50, 열마다 가장 높은 값 초록)</h2>
<table><thead>
<tr><th rowspan="2">버전</th><th colspan="4">킥보드 test 68장 (이미지 축소 배율)</th><th colspan="6">v4 검증셋 1,055장 클래스별 (5클래스 모델만)</th></tr>
<tr>{"".join(f"<th>{k}</th>" for k in KICK)}{"".join(f"<th>{c}</th>" for c in CLS_KO)}</tr>
</thead><tbody>{rows_res}</tbody></table>

<h2>3. 읽는 법 · 주의</h2>
<ul>
<li><b>킥보드 test</b>: 모든 버전이 학습에 쓰지 않은 같은 68장. r0.5/r0.25/r0.125는 이미지를 줄여 멀리 있는 작은 킥보드를 흉내 낸 것.</li>
<li><b>v4 검증셋</b>: 킥보드·배경·야생동물·COCO 동물·COCO 오토바이. v1·v2는 킥보드만 있는 모델이라 빈칸. v3는 COCO 오토바이 사진을 학습하지 않아 오토바이 점수가 불리함.</li>
<li>사람·차 정답 일부는 yolo11m이 붙인 의사 라벨이라 실제 정확도와 차이가 있을 수 있음.</li>
{V5_NOTE}
<li>영상 테스트(v3→v4, 14분 서울 도로주행): 오토바이 131→211개 탐지, 사람을 동물로 잡던 오검출 감소. 단 킥보드에 오토바이 박스가 겹치는 장면 발견.</li>
</ul>
</html>"""
Path(sys.argv[2]).write_text(html, encoding="utf-8")
print("saved", sys.argv[2])
