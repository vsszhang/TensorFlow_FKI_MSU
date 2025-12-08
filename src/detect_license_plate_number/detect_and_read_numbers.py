import cv2
import numpy as np
from minivan_model import load_minivan_model, predict_patch

BASE = "src/detect_license_plate_number/"

def find_minivan_model_in_image(
    img_bgr,
    minivan_medel,
    window_size=128,
    stride=64,
    prob_thr=0.8          # probobility threshold of 'minivan'
):
    H, W = img_bgr.shape[:2]
    candidates = []

    for y in range(0, H - window_size + 1, stride):
        for x in range(0, W - window_size + 1, stride):
            patch = img_bgr[y:y+window_size, x:x+window_size]

            # classify patch
            label, prob = predict_patch(minivan_medel, patch)

            # target and pend minivan
            if label == "minivan" and prob >= prob_thr:
                candidates.append((x, y, x + window_size, y + window_size, prob))
    print(f"[INFO] Number of candidate minivan: {len(candidates)}")
    return candidates

def main():
    # load model
    model = load_minivan_model(BASE + "model/minivan_cnn.keras")

    # read a test image
    img = cv2.imread(BASE + "data/images/minivan2.jpg")
    candidates = find_minivan_model_in_image(img, model, window_size=128, stride=64, prob_thr=0.8)

    # draw the candidate frame of minivan
    vis = img.copy()
    for x1, y1, x2, y2, prob in candidates:
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(vis, f"{prob:.2f}", (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imwrite(BASE + "output/minivan_candidates.png", vis)
    print("[SAVE] Save the cadidate frame into src/../output/minivan_candidates.png")


if __name__ == "__main__":
    main()