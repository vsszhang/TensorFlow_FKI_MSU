import cv2
import numpy as np
from minivan_model import load_minivan_model, predict_patch

BASE = "src/detect_license_plate_number/"


def find_minivan_model_in_image(
    img_bgr,
    minivan_medel,
    window_size=128,
    stride=64,
    prob_thr=0.96,  # probobility threshold of 'minivan'
):
    H, W = img_bgr.shape[:2]
    candidates = []

    for y in range(0, H - window_size + 1, stride):
        for x in range(0, W - window_size + 1, stride):
            patch = img_bgr[y : y + window_size, x : x + window_size]

            # classify patch
            label, prob = predict_patch(minivan_medel, patch)

            # target and pend minivan
            if label == "minivan" and prob >= prob_thr:
                candidates.append((x, y, x + window_size, y + window_size, prob))
    print(f"[INFO] Number of candidate minivan: {len(candidates)}")
    return candidates


def nms_boxes(candidates, iou_thr=0.3):
    """Merge overlapping frame

    Args:
        candidates (any): patch candidates
        iou_thr (float, optional): IOU threshold. Defaults to 0.3.

    Returns:
        list: list of merged frame
    """
    if not candidates:
        return []

    # transfer to numpy
    boxes = np.array(
        [[x1, y1, x2, y2] for (x1, y1, x2, y2, p) in candidates], dtype=float
    )
    scores = np.array([p for (_, _, _, _, p) in candidates], dtype=float)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # Sort by confidence level from highest to lowest
    order = scores.argsort()[::-1]

    keep_indices = []

    while order.size > 0:
        i = order[0]  # current frame with most highest confidence level
        keep_indices.append(i)

        # Other frame makes IoU with current one
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        # remain the not overlapping part
        remain = np.where(iou < iou_thr)[0]
        order = order[remain + 1]

    # merged frame of candidates
    merged = [candidates[i] for i in keep_indices]
    return merged


def find_digit_candidates(roi_bgr, debug_prefix=None):
    """Find a rectangle included with number in a minivan ROI

    Args:
        roi_bgr (MatLike): ROI's bgr
        debug_prefix (Any, optional): debug triger. Defaults to None.

    Returns:
        list: list of rectangle included with number
    """
    roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    # 1. light Gaussian Bulr
    roi_bulr = cv2.GaussianBlur(roi_gray, (5, 5), 0)

    # 2. OTUS auto threshold. get black-white image
    _, roi_bin = cv2.threshold(
        roi_bulr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # full the hole and connect strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    roi_morph = cv2.morphologyEx(roi_bin, cv2.MORPH_CLOSE, kernel, iterations=1)

    if debug_prefix is not None:
        cv2.imwrite(f"{BASE}output/{debug_prefix}_gray.png", roi_gray)
        cv2.imwrite(f"{BASE}output/{debug_prefix}_bin.png", roi_bin)
        cv2.imwrite(f"{BASE}output/{debug_prefix}_morph.png", roi_morph)

    # 4. find frame
    contours, _ = cv2.findContours(
        roi_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    H, W = roi_gray.shape[:2]
    digit_boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h

        # filter
        if area < 0.0005 * W * H:
            continue
        if area > 0.3 * W * H:
            continue

        aspect = w / float(h + 1e-6)

        if aspect < 0.2 or aspect > 0.6:
            continue

        if h < 0.03 * H:
            continue

        digit_boxes.append((x, y, x + w, y + h))

    return digit_boxes


def main():
    # step 1: load model
    model = load_minivan_model(BASE + "model/minivan_cnn.keras")

    # step 2: read a test image
    img = cv2.imread(BASE + "data/images/minivan2.jpg")

    # step 3: sliding window find minivan frame
    candidates = find_minivan_model_in_image(
        img, model, window_size=128, stride=64, prob_thr=0.8
    )
    print(f"[INFO] raw candidates: {len(candidates)}")

    # step 4: NMS merge for candidates
    merged = nms_boxes(candidates, iou_thr=0.3)
    print(f"[INFO] after NMS: {len(merged)}")

    # step 5: draw the candidate frame of minivan
    vis = img.copy()
    for x1, y1, x2, y2, prob in candidates:
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.putText(
            vis,
            f"{prob:.2f}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1,
        )

    for x1, y1, x2, y2, prob in merged:
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(
            vis,
            f"{prob:.2f}",
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    if merged:
        x1, y1, x2, y2, prob = merged[0]
        roi = img[y1:y2, x1:x2]

        digit_boxes = find_digit_candidates(roi, debug_prefix="debug_minivan2")

        # draw the digit candidates
        for dx1, dy1, dx2, dy2 in digit_boxes:
            cv2.rectangle(
                vis, (x1 + dx1, y1 + dy1), (x1 + dx2, y1 + dy2), (255, 0, 0), 2
            )

    cv2.imwrite(BASE + "output/minivan_candidates.png", vis)
    print("[SAVE] Save the cadidate frame into src/../output/minivan_candidates.png")


if __name__ == "__main__":
    main()
