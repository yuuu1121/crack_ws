#!/usr/bin/env python3
"""웹캠 실시간 크랙 탐지 + 오탐 프레임 수집.

키: q 종료 | s 현재 프레임 저장(오탐 발견 시 → captures/, 나중에 negative로 재학습)
"""
import time
from pathlib import Path
import cv2
from ultralytics import YOLO

WEIGHTS = Path(__file__).parent.parent / 'models' / 'best.pt'
CAPTURES = Path(__file__).parent / 'captures'
CAPTURES.mkdir(exist_ok=True)
CONF = 0.4

model = YOLO(str(WEIGHTS))
cap = cv2.VideoCapture(0)
assert cap.isOpened(), "/dev/video0 열기 실패 — 컨테이너면 --device /dev/video0 확인"

while True:
    ok, frame = cap.read()
    if not ok:
        break
    r = model.predict(frame, conf=CONF, verbose=False)[0]
    vis = r.plot()
    cv2.putText(vis, f"crack: {len(r.boxes)}  [s]ave [q]uit", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('crack detect', vis)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    if k == ord('s'):
        p = CAPTURES / f"cap_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(str(p), frame)  # 박스 없는 원본 저장 (재학습용)
        print(f"saved {p}")

cap.release()
cv2.destroyAllWindows()
