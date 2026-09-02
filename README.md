# crack_ws

콘크리트 크랙 실시간 검출 (YOLOv8n). 줄눈·이음새 오탐을 negative 데이터 주입으로 제거한 모델 포함.

## 젯슨에서 바로 실행 (도커 없이)

```bash
git clone https://github.com/yuuu1121/crack_ws.git
cd crack_ws
bash demo/setup_jetson.sh          # ultralytics 설치 + CUDA 확인
python3 demo/camera_detect.py      # USB 카메라(/dev/video0) 실시간 탐지
```

데모 키: `q` 종료, `s` 현재 프레임을 `demo/captures/`에 저장(오탐 수집용).

- 가중치: `models/best.pt` (오탐 개선판 fixed2, 6MB)
- GPU가 안 잡히면(setup 스크립트가 알려줌) NVIDIA 젯슨용 torch wheel 설치 필요 — 스크립트 출력의 링크 참고. CPU로도 동작은 함.
- FPS 부족 시 젯슨 위에서: `yolo export model=models/best.pt format=engine half=True`

## 프로젝트 상태·이력·재학습 방법

`CLAUDE.md` 참고 (Claude Code로 이어서 작업 가능하도록 전체 맥락 기록됨).

## Attribution

Dataset: [Underwater Crack](https://universe.roboflow.com/s-workspace-500dw/underwater-crack-glkyb) (Roboflow Universe, CC BY 4.0) + 합성 negative 추가.
