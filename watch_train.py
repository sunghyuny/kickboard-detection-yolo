"""
학습 진행 상황 보기: 로그의 현재 진행바 + results.csv의 epoch별 성능을 5초마다 갱신

사용법 (학습 로그는 runs/kickboard/<name>_log.txt, 결과는 runs/kickboard/<name>/results.csv 라고 가정):
    python watch_train.py --name train_v2_s09
"""
import argparse
import csv
import os
import re
import time
from pathlib import Path

RUNS = Path(__file__).parent / "runs" / "kickboard"
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def current_line(log):
    """로그 끝에서 마지막 진행바 한 줄 (진행바는 줄바꿈 없이 \\r / ESC[K로 덮어씀)"""
    with open(log, "rb") as f:
        f.seek(max(0, log.stat().st_size - 4000))
        tail = ANSI.sub("\n", f.read().decode("utf-8", "ignore").replace("\r", "\n"))
    lines = [l.strip() for l in tail.split("\n") if l.strip()]
    return lines[-1] if lines else ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="runs/kickboard/<name>")
    parser.add_argument("--interval", type=float, default=5)
    args = parser.parse_args()
    log, res = RUNS / f"{args.name}_log.txt", RUNS / args.name / "results.csv"

    while True:
        out = [f"[{time.strftime('%H:%M:%S')}] {args.name}", ""]
        out.append("지금: " + (current_line(log) if log.exists() else "로그 없음"))
        if res.exists():
            with open(res, newline="") as f:
                rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]
            out += ["", "epoch   분   val mAP50   mAP50-95   recall   precision"]
            for r in rows:
                out.append(f"{int(float(r['epoch'])):5d} {float(r['time']) / 60:5.0f}   {float(r['metrics/mAP50(B)']):9.3f}"
                           f"   {float(r['metrics/mAP50-95(B)']):8.3f}   {float(r['metrics/recall(B)']):6.3f}"
                           f"   {float(r['metrics/precision(B)']):9.3f}")
        if "test mAP50" in log.read_text("utf-8", "ignore")[-3000:] if log.exists() else False:
            out += ["", "학습 끝남"]
        os.system("cls" if os.name == "nt" else "clear")
        print("\n".join(out), flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
