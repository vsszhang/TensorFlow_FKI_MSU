"""
Main logic of code:

file in the output/selected/       --->      parse into (x1, y1, x2, y2, score)
                                                      ↓
                                        NMS remove the overlapping image field
                                                      ↓
                                    Draw the final frame on the original image
"""

import re
import os
from pathlib import Path
import cv2
import numpy as np

ORIG_IMG = "data/input/car.jpg"
SELECTED_DIR = "output/selected"
OUT_IMG = "output/merged_result.png"

# IoU (Intersection over Union) threshold: the smaller value, the easier it is to be merged
IOU_THRESH = 0.35

# Type
CLASSES_KEEP = {"car", "truck"}
name_re = re.compile(
    r"(?P<x1>\d+)-(?P<x2>\d+)_(?P<y1>\d+)-(?P<y2>\d+)-(?P<label>[A-Za-z0-9_]+)-(?P<score>\d+\.\d+)\.png$"
)

'''
    Desc: Calcuate IoU
'''
def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    # Calculate coordinate random of two frames
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = max(ax2, bx2)
    inter_y2 = max(ay2, by2)

    # Merged frame's weight and height
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter + 1e-9

    return inter / union

'''
    Desc: Calculate NMS
    Input: [[x1,y1,x2,y2], ...], [0.8, 0.6, ...]
'''
def nms(bboxes, scores, iou_thr = 0.5):
    idxs = np.argsort(scores)[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        rest = idxs[1:]
        ious = np.array([iou_xyxy(bboxes[i], bboxes[j]) for j in rest])
        idxs = rest[ious < iou_thr]
    return keep

'''
    Step 1: Read original image
'''
img = cv2.imread(ORIG_IMG)
H, W = img.shape[:2]

'''
    Step 2: Read every fliterd file and parse their coordinate and trust value
'''
records = []
for p in Path(SELECTED_DIR).glob("*.png"):
    m = name_re.search(p.name)
    if not m:
        continue
    x1 = int(m.group("x1"))
    x2 = int(m.group("x2"))
    y1 = int(m.group("y1"))
    y2 = int(m.group("y2"))
    label = m.group("label")
    score = float(m.group("score"))

    # Only keep the targeted type
    if label not in CLASSES_KEEP:
        continue

    # Anti coordinate over the range of original image
    x1 = max(0, min(x1, W - 1))
    x2 = max(0, min(x2, W - 1))
    y1 = max(0, min(y1, H - 1))
    y2 = max(0, min(y2, H - 1))

    # Only keep the working frame
    if x2 > x1 and y2 > y1:
        records.append((label, score, [x1, y1, x2, y2]))

if not records:
    print("[INFO] No working frame in the directory selected/")
    exit(0)

'''
    Step 3: Group by type, execute NMS merge
'''
font = cv2.FONT_HERSHEY_SIMPLEX
color = (0, 0, 255)
thickness = 2
merged = []

for cls in sorted(CLASSES_KEEP):
    cls_boxes = [r[2] for r in records if r[0] == cls]
    cls_scores = [r[1] for r in records if r[0] == cls]
    if not cls_boxes:
        continue

    keep_idx = nms(cls_boxes, cls_scores, iou_thr=IOU_THRESH)
    for k in keep_idx:
        merged.append((cls, cls_scores[k], cls_boxes[k]))

'''
    Step 4: Draw the final merged frame on the original image
'''
vis = img.copy()
for label, score, (x1, y1, x2, y2) in merged:
    # Draw the rectangle
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
    text = f"{label} {score: .2f}"
    cv2.putText(vis, text, (x1, max(0, y1 - 6)), font, 0.7, color, 2)

'''
    Step 5: Save the result and output the statistics
'''
print(f"[MERGE] Number of slice: {len(records)} -> Number of merged frame: {len(merged)}")
# os.mkdir(Path(OUT_IMG).parent, exist_ok=True)
cv2.imwrite(OUT_IMG, vis)
print(f"[SAVE] Result: {OUT_IMG}")