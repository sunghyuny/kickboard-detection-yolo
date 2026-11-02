"""
COCO 사전학습 YOLO(차 / 오토바이) + 킥보드 파인튜닝 모델을 같이 돌려서 영상 디텍션

프레임마다 따로 디텍션하면 박스가 깜빡이므로
ByteTrack으로 추적 + 잠깐 놓친 박스 유지 + 박스 위치 평활화를 한다

사용법:
    python detect_video.py --source test_videos/mixkit_40692.mp4
    python detect_video.py --source test_videos --no-show     # 폴더 안 영상 일괄 처리
"""
import argparse
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

HERE = Path(__file__).parent
# COCO 클래스 id: 0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck
COCO_CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
COLORS = {"car": (255, 128, 0), "bus": (255, 0, 255), "truck": (128, 0, 255), "motorcycle": (0, 128, 255),
          "bicycle": (0, 255, 255), "person": (200, 200, 200), "kickboard": (0, 255, 0), "animal": (0, 0, 255)}
MAX_AREA = 0.5  # 화면 절반 이상을 덮는 박스는 오검출로 보고 버림
VEHICLES = {"car", "bus", "truck"}
INSIDE = 0.5  # 킥보드 박스가 이 비율 이상 차/버스/트럭 박스 안에 있으면 오검출로 보고 버림


def inside_ratio(a, b):
    """박스 a 면적 중 박스 b와 겹치는 비율"""
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return iw * ih / max((a[2] - a[0]) * (a[3] - a[1]), 1)


class Smoother:
    """track id별로 박스를 지수평균하고, 놓친 뒤에도 hold 프레임 동안 유지"""

    def __init__(self, alpha, hold):
        self.alpha, self.hold = alpha, hold
        self.tracks = {}  # id -> [box, name, conf, 놓친 프레임 수]

    def update(self, dets):
        for t in self.tracks.values():
            t[3] += 1
        for tid, box, name, conf in dets:
            if tid in self.tracks:
                old = self.tracks[tid][0]
                box = [self.alpha * b + (1 - self.alpha) * o for b, o in zip(box, old)]
            self.tracks[tid] = [box, name, conf, 0]
        self.tracks = {k: t for k, t in self.tracks.items() if t[3] <= self.hold}
        return self.tracks


def tracked(model, frame, names, args, key):
    # conf를 낮게 줘야 ByteTrack이 점수 낮은 프레임에서도 기존 트랙을 이어붙임
    kw = {"classes": list(names)} if names is COCO_CLASSES else {}
    r = model.track(frame, persist=True, conf=0.1, tracker="bytetrack.yaml", verbose=False, **kw)[0]
    if r.boxes.id is None:
        return []
    h, w = frame.shape[:2]
    dets = []
    for box, cls, conf, tid in zip(r.boxes.xyxy.tolist(), r.boxes.cls.int().tolist(),
                                   r.boxes.conf.tolist(), r.boxes.id.int().tolist()):
        x1, y1, x2, y2 = box
        if (x2 - x1) * (y2 - y1) > MAX_AREA * w * h:
            continue
        dets.append(((key, tid), box, names[cls], conf))
    return dets


def run(path, coco, kick, args):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = Path(args.out) / f"{path.stem}_det.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # 영상마다 트래커 초기화 (persist=True라 이전 영상 트랙이 남지 않게)
    for m in (coco, kick):
        if m is not None and m.predictor is not None:
            m.predictor.trackers = None
            del m.predictor.trackers
    smoother = Smoother(args.alpha, args.hold)
    s = w / 1280  # 해상도에 맞춰 선/글자 크기 조절 (4K에서 너무 얇아지지 않게)
    thick, font = max(2, round(2 * s)), 0.6 * s
    seen = {name: set() for name in COLORS}  # 영상 전체에서 등장한 트랙 id
    streak = Counter()  # 킥보드 트랙 id -> kick_low 이상이 이어진 프레임 수
    kick_frames = 0  # 킥보드가 하나라도 표시된 프레임 수

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        coco_dets = tracked(coco, frame, COCO_CLASSES, args, "coco") if coco else []
        kick_all = tracked(kick, frame, kick.names, args, "kick")  # 5클래스 모델이면 차·사람 등도 여기서 나옴
        vehicles = [d[1] for d in coco_dets + kick_all if d[2] in VEHICLES and d[3] >= args.conf]
        kick_dets = [d for d in kick_all if d[2] != "kickboard" or all(inside_ratio(d[1], v) < INSIDE for v in vehicles)]
        dets = coco_dets + kick_dets
        # 멀리 있는 킥보드는 점수가 낮게(0.35~0.5) 꾸준히 나오고 헛것은 한두 프레임만 깜빡이므로, 낮은 점수도 연속이면 인정
        streak = Counter({d[0]: streak[d[0]] + 1 for d in kick_dets if d[2] == "kickboard" and d[3] >= args.kick_low})
        # 표시는 conf 기준을 넘긴 적이 있는 트랙만 (이미 표시 중이면 점수가 내려가도 유지)
        dets = [d for d in dets if d[3] >= (args.kick_conf if d[2] == "kickboard" else args.conf)
                or d[0] in smoother.tracks or 0 < args.kick_streak <= streak[d[0]]]
        tracks = smoother.update(dets)

        counts = Counter()
        for (_, tid), (box, name, conf, _) in tracks.items():
            x1, y1, x2, y2 = map(int, box)
            counts[name] += 1
            seen[name].add(tid)
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[name], thick)
            cv2.putText(frame, f"{name} #{tid}", (x1, max(y1 - int(6 * s), int(12 * s))),
                        cv2.FONT_HERSHEY_SIMPLEX, font, COLORS[name], thick)

        kick_frames += counts["kickboard"] > 0
        y = int(30 * s)
        for name, color in COLORS.items():
            cv2.putText(frame, f"{name}: {counts[name]}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, font * 1.3, color, thick)
            y += int(30 * s)
        writer.write(frame)

        if not args.no_show:
            cv2.imshow("car / motorcycle / kickboard", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    print(f"{path.name}: 등장한 개수 {({n: len(s) for n, s in seen.items()})}  킥보드 표시 프레임 {kick_frames}  ->  {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="영상 파일 또는 영상 폴더")
    parser.add_argument("--coco", default=str(HERE / "yolo11s.pt"), help="차/오토바이용 COCO 가중치 (\"\"이면 끄고 --kick 모델 하나만 사용)")
    parser.add_argument("--kick", default=str(HERE / "runs/kickboard/train_e100/weights/best.pt"))
    parser.add_argument("--conf", type=float, default=0.4, help="새 박스를 표시하기 시작하는 점수")
    parser.add_argument("--kick-conf", type=float, default=0.6, help="킥보드 전용 기준 (도로 배경 오검출이 많아 더 높게)")
    parser.add_argument("--kick-low", type=float, default=0.35, help="연속 확인용 킥보드 낮은 기준")
    parser.add_argument("--kick-streak", type=int, default=5, help="kick-low 이상이 이 프레임 수만큼 이어지면 표시 (0이면 끔)")
    parser.add_argument("--hold", type=int, default=8, help="놓친 박스를 유지할 프레임 수")
    parser.add_argument("--alpha", type=float, default=0.4, help="박스 평활화 (작을수록 부드럽고 느리게 따라감)")
    parser.add_argument("--no-show", action="store_true", help="창 없이 결과 영상만 저장")
    parser.add_argument("--out", default=str(HERE / "runs" / "video"), help="결과 영상 저장 폴더")
    args = parser.parse_args()

    coco, kick = (YOLO(args.coco) if args.coco else None), YOLO(args.kick)
    src = Path(args.source)
    videos = sorted(src.glob("*.mp4")) if src.is_dir() else [src]
    for v in videos:
        run(v, coco, kick, args)
    if not args.no_show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
