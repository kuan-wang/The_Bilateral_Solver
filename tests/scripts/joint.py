import cv2
import numpy as np

joint = cv2.imread("../build/confidence.png")
src = cv2.imread("../build/reference.png")
dst = cv2.imread("../build/target.png")
#cv2.jointBilateralFilter(joint, src, dst, d, sigmaColor, sigmaSpace[, borderType])
#cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace[, dst[, borderType]])
d = 32
sigmaColor = 4
sigmaSpace = 8
tar = cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace)
print tar.shape
cv2.imshow("src",src)
cv2.imshow("tar",tar)
tar1 = tar - src
cv2.imshow("tar1",tar1)
# cv2.jointBilateralFilter(joint, src, dst, d, sigmaColor, sigmaSpace)
# # tar1 = tar - src
# cv2.imshow("dst",dst)

cv2.waitKey(0)
