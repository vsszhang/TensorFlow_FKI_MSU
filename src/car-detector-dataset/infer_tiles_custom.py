"""
Splited block resoning, filter and save
"""
from pathlib import Path
import re
import numpy as np
import cv2
from tqdm import tqdm
from tensorflow import keras
import argparse

MODEL_PATH = Path("models/custom_dataset_cnn.keras")
CLASSES_TXT = MODEL_PATH.with_suffix(".classes.txt")

with CLASSES_TXT.open("r", encoding="utf-8") as f:
    CLASS_NAMES = [line.strip() for line in f if line.strip()]

print("Loaded classes:", CLASS_NAMES)
TARGETS = {"minivan"}

""" Load traning model """
def load_model(model_path: str):
    model = keras.models.load_model(model_path)
    return model

""" Preprocess batch filtered img 
    BGR -> RGB -> 32 * 32 -> [0, 1], return (N, 32, 32, 3)
"""
def preprocess_batch(img_paths):
    batch = []
    raws = []
    for p in img_paths:
        bgr = cv2.imread(str(p))
        if bgr is None:
            batch.append(None)
            raws.append(None)
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (32, 32), interpolation=cv2.INTER_AREA)
        batch.append(rgb.astype("float32"))
        raws.append(bgr)
    valid_imgs = [im for im in batch if im is not None]
    x = np.stack(valid_imgs, axis=0) if valid_imgs else np.empty((0, 32, 32, 3))
    return x, raws

""" Parse filterd image, according to the file name
"""
def parse_coords(stem: str):
    m = re.match(r"(\d+)-(\d+)_([\d]+)-(\d+)-", stem)
    if not m:
        return None
    x1, x2, y1, y2 = map(int, m.groups())
    return x1, x2, y1, y2

def main(
        split_dir: str,
        out_dir: str,
        model_path: str,
        threshold: float,
        batch_size: int
    ):
    split_dir = Path(split_dir)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(model_path)
    tiles = sorted(split_dir.glob("*.png"))
    if not tiles:
        print(f"[WARN] no tiles found in {split_dir}"); return
    
    kept = 0
    for i in tqdm(range(0, len(tiles), batch_size), desc="infer"):
        batch_paths = tiles[i:i+batch_size]
        x, raws = preprocess_batch(batch_paths)
        if x.shape[0] == 0:
            continue
        probs = model.predict(x, verbose=0)

        j = 0
        for p in batch_paths:
            bgr = raws[j]; j += 1
            if bgr is None:
                continue
            pr = probs[j-1]
            idx = int(np.argmax(pr))
            label = CLASS_NAMES[idx]
            conf = float(pr[idx])

            if label in TARGETS and conf >= threshold:
                coords = parse_coords(p.stem)
                if coords is not None:
                    x1, x2, y1, y2 = coords
                    save_name = f"{x1}-{x2}_{y1}-{y2}-{label}-{conf:.2f}.png"
                else:
                    save_name = p.name.replace("-none.png", f"-{label}.png")
                cv2.imwrite(str(out_dir / save_name), bgr)
                kept += 1

        print(f"[KEEP] {kept} tiles saved to {out_dir.resolve()} (thr={threshold})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-dir", default="output/split", help="Director of splited block")
    ap.add_argument("--out-dir", default="output/selected_custom", help="Director of saving after filter")
    ap.add_argument("--model", default="models/custom_dataset_cnn.keras", help="Model path")
    ap.add_argument("--thr", type=float, default=0.90, help="threshold of confidence")
    ap.add_argument("--bs", type=int, default=256, help="batch size")
    args = ap.parse_args()
    main(args.split_dir, args.out_dir, args.model, args.thr, args.bs), 
