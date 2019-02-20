import numpy as np
import cv2
UBIT='srujanko'
np.random.seed(sum([ord(c) for c in "UBIT"]))
from random import*
import random

image1= cv2.imread('D://mountain1.jpg')
image2= cv2.imread('D://mountain2.jpg')
img1 = cv2.imread('D://mountain1.jpg',0) # queryImage
img2 = cv2.imread('D://mountain2.jpg',0) # trainImage

# Initiate SIFT detector
sift = cv2.cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
cv2.imwrite("D://task_1sift1.jpg",cv2.drawKeypoints(img1, kp1,image1.copy()))
cv2.imwrite("D://task_1sift2.jpg",cv2.drawKeypoints(img2, kp2,image2.copy()))
# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)
best_match = []
x=[]
for m,n in matches:
    if m.distance < 0.75*n.distance:
        best_match.append([m])
        x.append(m)
cv2.imwrite("D://task1_matches_knn.jpg",cv2.drawMatchesKnn(image1,kp1,image2,kp2,best_match,None,flags=2))
src_pts = np.float32([ kp1[m.queryIdx].pt for m in x ]).reshape(-1,1,2)
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in x ]).reshape(-1,1,2)

M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
print(M)
matchesMaskAll = []    
matchesMask = mask.ravel().tolist()
masko = np.asarray(matchesMask)
dnd = np.where(masko==1)
dnd = np.asarray(dnd).ravel()
msk =[]
seed(sum([ord(c) for c in UBIT]))
dnd=np.random.choice(dnd, 10, replace=False)
msk =np.zeros(len(matchesMask))
for i in range(len(dnd)):
    msk[dnd[i]]=1

h, w = img1.shape
pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)



draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = msk, # draw only inliers
                   flags = 2)


img10 = cv2.drawMatches(image1,kp1,image2,kp2,x,None,**draw_params)

cv2.imwrite("D://task1_matches.jpg",img10)
def warpTwoImages(img1, img2, H):
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    pts1 =np.float64([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float64([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts =np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate

    result = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))
    result[t[1]:h1+t[1],t[0]:w1+t[0]] = img1
    return result

result = warpTwoImages(img2, img1, M)
cv2.imwrite("D://task1 pano.jpg",result)