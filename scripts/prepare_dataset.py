#!/usr/bin/env python3
"""COCO -> YOLO 변환 + pier hard-negative 주입 + 수중 틴트 증강.

산출:
  dataset_yolo_baseline/  크랙 데이터만 (비교용 baseline)
  dataset_yolo_fixed/     크랙 + pier negatives + 수중 틴트 증강본
  eval_negatives/         학습에 안 쓴 pier 이미지 50장 (오탐 평가 전용)
"""
import json, random, shutil
from pathlib import Path
from PIL import Image, ImageEnhance

WS = Path('/root/home/crack_ws')
COCO = WS / 'dataset_coco'
PIER_RAW = Path('/root/home/vlm_ws/src/stonefish_sim/stonefish_vlm/pier_detection/dataset/raw')
random.seed(0)


def convert_split(coco_dir, split, out_root):
    d = json.load(open(coco_dir / split / '_annotations.coco.json'))
    img_dir = out_root / 'images' / split
    lbl_dir = out_root / 'labels' / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    anns_by_img = {}
    for a in d['annotations']:
        anns_by_img.setdefault(a['image_id'], []).append(a)
    for im in d['images']:
        src = coco_dir / split / im['file_name']
        shutil.copy(src, img_dir / im['file_name'])
        lines = []
        for a in anns_by_img.get(im['id'], []):
            x, y, w, h = a['bbox']
            cx, cy = (x + w / 2) / im['width'], (y + h / 2) / im['height']
            lines.append(f"0 {cx:.6f} {cy:.6f} {w / im['width']:.6f} {h / im['height']:.6f}")
        (lbl_dir / (Path(im['file_name']).stem + '.txt')).write_text('\n'.join(lines))
    return len(d['images'])


def underwater_tint(src, dst):
    im = Image.open(src).convert('RGB')
    r, g, b = im.split()
    # 청록 캐스트: R 감쇠, G/B 유지 — 수중에서 적색이 먼저 흡수됨
    r = r.point(lambda v: int(v * random.uniform(0.35, 0.6)))
    g = g.point(lambda v: int(v * random.uniform(0.85, 1.0)))
    im = Image.merge('RGB', (r, g, b))
    im = ImageEnhance.Brightness(im).enhance(random.uniform(0.5, 0.8))
    im = ImageEnhance.Contrast(im).enhance(random.uniform(0.7, 0.95))
    im.save(dst, quality=90)


def write_yaml(root, name):
    (root / 'data.yaml').write_text(
        f"path: {root}\ntrain: images/train\nval: images/valid\ntest: images/test\nnames:\n  0: crack\n")


def main():
    # 1) baseline: 단순 변환
    base = WS / 'dataset_yolo_baseline'
    fixed = WS / 'dataset_yolo_fixed'
    for root in (base, fixed):
        if root.exists():
            shutil.rmtree(root)
        for split in ('train', 'valid', 'test'):
            n = convert_split(COCO, split, root)
            print(f"{root.name}/{split}: {n} imgs")
        write_yaml(root, root.name)

    # 2) fixed: 수중 틴트 증강 (train 크랙 이미지 전체 1x)
    tr_img = fixed / 'images' / 'train'
    tr_lbl = fixed / 'labels' / 'train'
    n_aug = 0
    for img in sorted(tr_img.glob('*.jpg')):
        lbl = tr_lbl / (img.stem + '.txt')
        if not lbl.read_text().strip():
            continue  # 배경엔 틴트 불필요 (pier negatives가 그 역할)
        underwater_tint(img, tr_img / f"uw_{img.name}")
        shutil.copy(lbl, tr_lbl / f"uw_{img.stem}.txt")
        n_aug += 1
    print(f"underwater-tint augmented: {n_aug}")

    # 3) pier hard negatives: train 120 / valid 30 / eval 홀드아웃 50
    pier = sorted(PIER_RAW.glob('*.jpg'))
    random.shuffle(pier)
    groups = {'train': pier[:120], 'valid': pier[120:150]}
    for split, files in groups.items():
        for f in files:
            shutil.copy(f, fixed / 'images' / split / f"neg_{f.name}")
            (fixed / 'labels' / split / f"neg_{f.stem}.txt").write_text('')
        print(f"pier negatives -> {split}: {len(files)}")
    ev = WS / 'eval_negatives'
    if ev.exists():
        shutil.rmtree(ev)
    ev.mkdir()
    for f in pier[150:200]:
        shutil.copy(f, ev / f.name)
    print(f"held-out eval negatives: {len(list(ev.glob('*.jpg')))}")


if __name__ == '__main__':
    main()
