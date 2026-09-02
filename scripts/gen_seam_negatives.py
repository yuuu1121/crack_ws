#!/usr/bin/env python3
"""'살짝의 틈'(줄눈·이음새) 합성 negative 생성 — 직선 어두운 선은 크랙이 아님을 가르친다.

콘크리트풍 노이즈 텍스처 위에 직선 seam(단일/격자)을 그린다.
크랙(불규칙 지그재그)과 달리 seam은 곧고 규칙적 — 이 구분이 학습 목표.
"""
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from pathlib import Path

WS = Path('/root/home/crack_ws')
random.seed(1)
np.random.seed(1)


def concrete_canvas(size=224):
    base = random.randint(120, 200)
    arr = np.clip(base + np.random.randn(size, size) * 8, 0, 255).astype(np.uint8)
    im = Image.fromarray(arr, 'L').filter(ImageFilter.GaussianBlur(0.6))
    # 점박이 (원본 데이터셋의 골재 반점 모사)
    d = ImageDraw.Draw(im)
    for _ in range(random.randint(10, 40)):
        x, y = random.randint(0, size), random.randint(0, size)
        r = random.randint(1, 3)
        d.ellipse([x, y, x + r, y + r], fill=random.randint(40, 100))
    return im.convert('RGB')


def draw_seam(im):
    """직선 틈 1~3개: 수직/수평/약간 기운 직선, 어두운 색."""
    d = ImageDraw.Draw(im)
    w, h = im.size
    for _ in range(random.randint(1, 3)):
        dark = random.randint(30, 90)
        width = random.randint(1, 4)
        if random.random() < 0.5:  # 수직 계열
            x = random.randint(10, w - 10)
            tilt = random.randint(-15, 15)
            d.line([(x, 0), (x + tilt, h)], fill=(dark, dark, dark), width=width)
        else:  # 수평 계열
            y = random.randint(10, h - 10)
            tilt = random.randint(-15, 15)
            d.line([(0, y), (w, y + tilt)], fill=(dark, dark, dark), width=width)
    return im.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 0.8)))


def main():
    ev = WS / 'eval_seams'
    ev.mkdir(exist_ok=True)
    for i in range(50):
        draw_seam(concrete_canvas()).save(ev / f'seam_{i:03d}.jpg', quality=90)
    print(f"eval seam negatives: 50 -> {ev}")

    # 학습용 seam negative (평가셋과 시드 이어져 있으나 랜덤 생성이라 이미지는 상이)
    tr = WS / 'train_seams'
    tr.mkdir(exist_ok=True)
    for i in range(120):
        draw_seam(concrete_canvas()).save(tr / f'seam_tr_{i:03d}.jpg', quality=90)
    print(f"train seam negatives: 120 -> {tr}")


if __name__ == '__main__':
    main()
