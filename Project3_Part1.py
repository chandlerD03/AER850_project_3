import cv2
import numpy as np
from matplotlib import pyplot as plt


img_real = cv2.imread("motherboard_image.jpeg", cv2.IMREAD_COLOR)
img_real = cv2.rotate(img_real, cv2.ROTATE_90_CLOCKWISE)  


img_blurred = cv2.blur(img_real, (15, 15))  

img_gray = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2GRAY)

img_thresholded = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 5)

edges = cv2.Canny(img_thresholded, 30, 150)  
edges_dilated = cv2.dilate(edges, None, iterations=8)  


contours, _ = cv2.findContours(edges_dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

mask = np.zeros_like(img_real)

largest_contour = max(contours, key=cv2.contourArea, default=None) 

cv2.drawContours(mask, [largest_contour], -1, (255, 255, 255), thickness=cv2.FILLED)

masked_img = cv2.bitwise_and(img_real, mask)



plt.figure(figsize=(16,16))
plt.imshow(edges, cmap="gray")
plt.axis("off")
plt.title("Edge Detection")

plt.figure(figsize=(16,16))
plt.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Mask")

plt.figure(figsize=(16, 16)) 
plt.imshow(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Masked Image")
plt.show()


print("Original Image Shape:", img_real.shape)
print("Masked Image Shape:", masked_img.shape)