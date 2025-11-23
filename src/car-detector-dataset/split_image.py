from pathlib import Path
import cv2
from tqdm import tqdm

"""
(title_w, title_h, offset_x, offset_y)
"""
SPECS = [
    (240, 240, 0, 0),
    (240, 240, 120, 120),
    (120, 120, 0, 0),
    (120, 120, 40, 40),
    (120, 120, 80, 80),
    ( 60, 60, 0, 0),
    ( 60, 60, 30, 30),
]

def slide_coords(W, H, tw, th, ox, oy):
    # sx = ox if ox > 0 else tw
    # sy = oy if oy > 0 else th
    # y = 0
    # while y + th <= H:
    #     x = 0
    #     while x + tw <= W:
    #         yield x, y
    #         x += sx
    #     y += sy
    y = oy
    while y + th <= H:
        x = ox
        while x + tw <= W:
            yield x, y
            x += tw
        y += th

def split_and_save(input_path="src/car-detector-dataset/data/input/car.jpg", out_dir="output/split"):
    print("TEST")

    # Defining output path, if not, create it
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(input_path))
    H, W = img.shape[:2]

    total = 0
    for tw, th, ox, oy in SPECS:
        coords = list(slide_coords(W, H, tw, th, ox, oy))
        desc = f"{tw}*{th}@{ox}@{oy}"
        for x, y in tqdm(coords, desc=desc):
            x1, y1 = x, y
            x2, y2 = x + tw, y + th
            roi = img[y1:y2, x1:x2]
            name = f"{x1}-{x2}_{y1}-{y2}-none.png"
            cv2.imwrite(str(out / name), roi)
            total +=   1

    print(f"[SPILT] Saved {total} tiles to {out.resolve()}")

if __name__ == "__main__":
    split_and_save()