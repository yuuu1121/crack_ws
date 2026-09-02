# crack_ws — 수중 크랙 YOLO 검출

콘크리트 크랙 검출 모델. 핵심 이력: **오탐(크랙 아닌 틈을 크랙으로 검출) 문제를 데이터 재구성으로 해결한 상태**이며, 다음 단계는 젯슨 실기기에서의 active learning 루프다.

## 지금까지 진행된 것 (2026-09-02, PC/RTX 4080에서 수행)

1. Roboflow "Underwater Crack" COCO 데이터셋(609장, 224×224 회색 콘크리트 클로즈업)을 YOLO 포맷으로 변환 — `scripts/prepare_dataset.py`
2. **오탐 원인 진단**: train 430장 중 배경(no-crack) 이미지가 12장뿐 → 모델이 "크랙 없음"을 학습 못 함. 오탐 유발원은 시뮬레이션 화면이 아니라 **직선 형태의 틈(줄눈·이음새)** — 합성 seam 이미지 50장으로 재현(baseline 44/50장 오탐, 91박스)
3. **처방**: ① 합성 seam negative 150장(빈 라벨) 주입 — `scripts/gen_seam_negatives.py` ② 수중 틴트 증강 418장 ③ pier 시뮬 negative 150장 → `dataset_yolo_fixed/`
4. yolov8n 학습 3종 비교 (imgsz 640, epochs 100, conf 0.25 평가):

| 모델 | seam 오탐(50장) | crack mAP50 | recall |
|---|---|---|---|
| baseline (원본만) | 91박스/44장 | 0.734 | 0.713 |
| fixed (틴트+pier neg) | 67박스/40장 | 0.711 | 0.674 |
| **fixed2 (최종 = models/best.pt)** | **0박스/0장** | 0.707 | 0.702 |

평가 스크립트: `scripts/eval_fp.py` (+ eval_seams/, eval_negatives/ 홀드아웃 — 학습에 미사용, 재학습 시에도 학습셋에 넣지 말 것)

## 다음 단계 (이어서 할 일)

- [ ] 젯슨 로컬(도커 X)에서 `demo/camera_detect.py` 실기 구동 확인 (USB캠 /dev/video0)
- [ ] 실카메라에서 새 유형 오탐 발생 시: 데모의 `s`키로 프레임을 `demo/captures/`에 수집 → PC로 가져가 빈 라벨 negative로 `dataset_yolo_fixed/`에 추가 → 재학습 (active learning 루프; 절차는 seam negative 주입과 동일)
- [ ] 젯슨 FPS 부족 시: 젯슨 위에서 `yolo export model=models/best.pt format=engine half=True` (TensorRT, .engine은 장치 종속이라 반드시 젯슨에서 변환)

## 환경 제약 (하드런 지식)

- **학습·val은 `workers=0` 필수** (PC 학습 컨테이너의 /dev/shm 64MB — DataLoader worker가 Bus error로 죽음. 도커에서 돌릴 땐 `--ipc=host`)
- 젯슨 로컬 실행: pip 기본 torch는 젯슨 GPU 미지원(CPU 폴백) — GPU 쓰려면 NVIDIA 젯슨용 torch wheel 필요 (`demo/setup_jetson.sh` 참고)
- CSI 카메라(리본)는 `cv2.VideoCapture(0)` 불가 — GStreamer `nvarguscamerasrc` 파이프라인 필요 (현재 데모는 USB캠 기준)
- 재학습은 PC(GPU)에서, 젯슨은 추론 전용. 재학습 명령: `dataset_yolo_fixed/data.yaml` 대상, hsv_h=0.05 hsv_v=0.6 degrees=10 (scripts/prepare_dataset.py 주석 및 git log 참고)

## 파일 지도

- `models/best.pt` — 배포 가중치 (fixed2). 젯슨은 이것만 씀
- `dataset_coco/` 원본 COCO / `dataset_yolo_fixed/` 최종 학습셋 / `dataset_yolo_baseline/` 비교용
- `scripts/` prepare_dataset.py(변환+증강+negative), gen_seam_negatives.py(seam 합성), eval_fp.py(오탐 비교)
- `demo/` camera_detect.py(실시간 데모+FP수집), setup_jetson.sh(젯슨 셋업), run_camera.sh(PC 도커용 — 젯슨에선 안 씀)
- `runs/`는 gitignore (학습 산출물, PC에만 존재)
