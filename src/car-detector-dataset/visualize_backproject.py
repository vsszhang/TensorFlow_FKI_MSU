"""
    1) Read image file from director output/selected/
    2) Parse image coordinates and restore the image position
    3) Merge overlapping block image
    4) Draw back merged block image to original image and save the viewing result
"""

from pathlib import Path
import re
import cv2
import numpy as np

""" Config path """
INPUT_IMG = "src/car-detector-dataset/data/input/car.jpg"
SELECTED_DIR = "src/car-detector-dataset/output/selected"
OUT_PATH = "src/car-detector-dataset/output/vis_result.png"

name_re = re.compile(r"(\d+)-(\d+)_([\d]+)-(\d+)-(.+)\.png")

"""
    Input: file name without extension
    Return: coordinate and label (x1, y1, x2, y2, label)
    
    Description: restore block images' coordinate and type label
"""
def parse_box(stem: str):
    m = name_re.match(stem + ".png")
    if not m:
        return None
    x1, x2, y1, y2, label = m.groups()
    return (int(x1), int(y1), int(x2), int(y2), label)

"""
    Description: Caculate IoU (Intersection over Union)
"""
def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]

    # Overlapping rectangle's upper left and lower right coordinate
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)

    # IoU = Intersection / Union
    return inter / (area_a + area_b - inter + 1e-9)

"""
    Description: Algorithm of union merge
                 merge blocks into one block, according to the IoU value
"""
def merge_boxes(boxes, iou_thr: float = 0.3):
    boxes = sorted(boxes, key=lambda b: (b[0], b[1]))
    merged = []
    used = [False] * len(boxes)

    for i in range(len(boxes)):
        if used[i]:
            continue

        # Seed block, index 'i'
        x1, y1, x2, y2, label = boxes[i]

        # Union block
        U = [x1, y1, x2, y2]
        used[i] = True

        for j in range(i + 1, len(boxes)):
            if used[j]:
                continue
            if iou(boxes[i], boxes[j]) >= iou_thr:
                # if overlapping rate is big, we can see this is one target
                # then merge to the Union block
                used[j] = True
                a, b, c, d, _ = boxes[j]
                U[0] = min(U[0], a) # min x1
                U[1] = min(U[1], b) # min y1
                U[2] = max(U[2], c) # max x2
                U[3] = max(U[3], d) # max y2
        merged.append((U[0], U[1], U[2], U[3], label))
    return merged

def main():
    # load the otiginal image
    img = cv2.imread(INPUT_IMG)
    H, W = img.shape[:2]

    sel_dir = Path(SELECTED_DIR)
    tiles = list(sel_dir.glob("*.png"))
    boxes = []

    for p in tiles:
        parsed = parse_box(p.stem)
        if parsed is None:
            continue
        x1, y1, x2, y2, label = parsed

        # make coordinates into the range
        x1 = int(np.clip(x1, 0, W))
        x2 = int(np.clip(x2, 0, W))
        y1 = int(np.clip(y1, 0, H))
        y2 = int(np.clip(y2, 0, H))

        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2, label))

    if not boxes:
        print("[INFO] no boxes found in selected/.")
        return
    
    # merge the box, threshold 0.3 is more esaier
    merged = merge_boxes(boxes, iou_thr=0.5)

    # visilation: draw the merged box into the original figure
    vis = img.copy()
    for (x1, y1, x2, y2, label) in merged:
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            vis, label, (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
        )

    Path("output").mkdir(exist_ok=True)
    cv2.imwrite(OUT_PATH, vis)
    print(f"[VIS] saved -> {OUT_PATH} | raw tiles: {len(boxes)} | merged: {len(merged)}")

if __name__ == "__main__":
    main()