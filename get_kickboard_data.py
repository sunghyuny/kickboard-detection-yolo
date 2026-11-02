"""
Roboflow Universe에서 킥보드 데이터셋(YOLO 포맷) 다운로드

데이터셋: https://universe.roboflow.com/dklyolo1/kickboard-dfkjo (v6, 1,591장)

사용법 (API 키는 코드에 쓰지 말고 환경변수로):
    PowerShell:  $env:ROBOFLOW_API_KEY="키"; python get_kickboard_data.py
"""
import os
from pathlib import Path

from roboflow import Roboflow

WORKSPACE = "dklyolo1"
PROJECT = "kickboard-dfkjo"
VERSION = 6
OUT_DIR = Path(__file__).parent / "datasets" / "kickboard"


def main():
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("ROBOFLOW_API_KEY 환경변수를 설정하세요")

    rf = Roboflow(api_key=api_key)
    version = rf.workspace(WORKSPACE).project(PROJECT).version(VERSION)
    # yolov11 포맷: images/labels + data.yaml
    dataset = version.download("yolov11", location=str(OUT_DIR))
    print("다운로드 위치:", dataset.location)


if __name__ == "__main__":
    main()
