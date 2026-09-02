#!/bin/bash
# 호스트(로컬 PC)에서 실행. CRACK_WS는 호스트 기준 crack_ws 경로로 수정.
set -e
CRACK_WS="${CRACK_WS:-$HOME/crack_ws}"

xhost +local:docker >/dev/null

docker run -it --rm \
  --gpus all \
  --device /dev/video0:/dev/video0 \
  --ipc=host \
  -e DISPLAY="$DISPLAY" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$CRACK_WS":/crack_ws \
  ultralytics/ultralytics \
  python3 /crack_ws/demo/camera_detect.py
