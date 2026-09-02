#!/bin/bash
# 젯슨 로컬(도커 X) 셋업. clone 후 1회 실행: bash demo/setup_jetson.sh
set -e

pip3 install --upgrade ultralytics opencv-python

echo
python3 - <<'EOF'
import torch
if torch.cuda.is_available():
    print(f"OK: CUDA 사용 가능 ({torch.cuda.get_device_name(0)})")
else:
    print("경고: torch가 CPU 전용입니다. yolov8n은 CPU로도 돌지만 느립니다.")
    print("젯슨 GPU를 쓰려면 pip 기본 torch가 아니라 NVIDIA 젯슨용 wheel이 필요합니다:")
    print("  JetPack 6: https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/")
    print("  (JetPack 버전 확인: cat /etc/nv_tegra_release)")
EOF

echo
echo "실행: python3 demo/camera_detect.py   (USB 카메라 /dev/video0 기준)"
