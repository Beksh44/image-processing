import cv2
import numpy as np
import matplotlib.pyplot as plt
import orb

img1 = cv2.imread('img/face1.jpg')
M = np.array([[1, abs(0.3), 0],[0,1,0]])
nW =  img1.shape[1] + abs(0.3 * img1.shape[0])
img2 = cv2.warpAffine(img1, M, (int(nW), img1.shape[0]))

kp1, scores1 = orb.detect_keypoints(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), threshold=20, border=10)
kp2, scores2 = orb.detect_keypoints(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), threshold=20, border=10)

kp1, kp2 = np.asarray(kp1), np.asarray(kp2)
fig, axs = plt.subplots(1, 2, figsize=(35, 10))
axs[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
sc1 = axs[0].scatter(kp1[:, 1], kp1[:, 0], c=scores1, cmap="Greens")
axs[0].axis("off")
axs[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
sc2 = axs[1].scatter(kp2[:, 1], kp2[:, 0], c=scores2, cmap="Greens")
axs[1].axis("off")
plt.colorbar(sc1, ax=axs[0])
plt.colorbar(sc2, ax=axs[1])
fig.suptitle("Features detected independently on a pair of images (colored by corresponding score)", size=26);