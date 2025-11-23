'''
    Desc: Read original picture
'''
from pathlib import Path
import cv2, matplotlib.pyplot as plt

img_path = Path("src/car-detector/data/input/car.jpg")
img = cv2.imread(str(img_path))

H, W = img.shape[:2]
print(f"Image size: {W}*{H}")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()