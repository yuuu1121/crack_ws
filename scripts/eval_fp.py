#!/usr/bin/env python3
"""홀드아웃 pier negative 50장(크랙 0개가 정답)에서 두 모델의 오탐 수 비교."""
from pathlib import Path
from ultralytics import YOLO

WS = Path('/root/home/crack_ws')
EVAL = sorted((WS / 'eval_negatives').glob('*.jpg'))
CONF = 0.25  # 배포 기본값과 동일 조건으로 비교

for name in ('baseline', 'fixed'):
    w = WS / 'runs' / name / 'weights' / 'best.pt'
    model = YOLO(str(w))
    results = model.predict(EVAL, conf=CONF, verbose=False)
    n_fp = sum(len(r.boxes) for r in results)
    n_img_fp = sum(1 for r in results if len(r.boxes))
    print(f"{name}: 오탐 박스 {n_fp}개, 오탐 발생 이미지 {n_img_fp}/{len(EVAL)}장")

    # 크랙 검출력 유지 확인: valid 크랙셋 재현율
    m = model.val(data=str(WS / 'dataset_yolo_baseline' / 'data.yaml'), split='val', verbose=False, workers=0)
    print(f"  crack valid: mAP50 {m.box.map50:.3f}, precision {m.box.mp:.3f}, recall {m.box.mr:.3f}")
