import numpy as np
import random
import cv2
np.random.seed(sum([ord(c) for c in "123"]))
from random import*
import random
UBIT='srujanko'
from matplotlib import pyplot as plt
def SIFTMATCH(img1,img2):
    img1=img1.copy()
    img2=img2.copy()
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    cv2.imwrite("D://task2_sift1.jpg",cv2.drawKeypoints(img1, kp1,img1.copy()))
    cv2.imwrite("D://task2_sift2.jpg",cv2.drawKeypoints(img2, kp2,img2.copy()))

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
#    matches = sorted(matches, key = lambda x:x.distance)
    good = []
    pts1 = []
    pts2 = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    cv2.imwrite("D://task2_matches_knn.jpg",cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2))
    
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.RANSAC)
    print(F)
  

# We select only inlier points
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]
    pts1en = list(enumerate(pts1, 1))
    pts2en = list(enumerate(pts2, 1))
    seed(sum([ord(c) for c in UBIT]))
    pts1enr = random.sample(pts1en, 10)
    
    pts3=[]
    pts4=[]
    ran = [i[0] for i in pts1enr]
    for x in ran:
        for y in pts1enr:
            if(x==y[0]):
                pts3.append(y[1])
    for x in ran:
        for y in pts2en:
            if(x==y[0]):
                pts4.append(y[1])
    pts3 = np.int32(pts3)
    pts4 = np.int32(pts4)

    
    
    pts1 = pts3
    pts2 = pts4
    colors = []   
    
    def drawlines(img1,img2,lines,pts1,pts2,colors):
        x=0
        ''' img1 - image on which we draw the epilines for the points in img2
            lines - corresponding epilines '''
        r,c = img1.shape
        img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
        img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
        
        for r,pt1,pt2 in zip(lines,pts1,pts2):
           
#            color = tuple(np.random.randint(0,255,3).tolist())
            x0,y0 = map(int, [0, -r[2]/r[1] ])
            x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
            img1 = cv2.line(img1, (x0,y0), (x1,y1), colors[x],1)
            img1 = cv2.circle(img1,tuple(pt1),5,colors[x],-1)
            img2 = cv2.circle(img2,tuple(pt2),5,colors[x],-1)
            x+=1
        return img1,img2
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)
    for i in range(10):
        seed(sum([ord(c) for c in UBIT]))
        colors.append(tuple(np.random.randint(0,255,3).tolist()))
        
    img5,img6 = drawlines(img1,img2,lines1,pts1,pts2,colors)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    img3,img4 = drawlines(img2,img1,lines2,pts2,pts1,colors)
    cv2.imwrite('D:\\task2_epi right.jpg',img3)
    cv2.imwrite('D:\\task2_epi left.jpg',img5)
 
    plt.show()
    
img1 = cv2.imread("D:\\tsucuba_left.PNG",0) 
img2 = cv2.imread("D:\\tsucuba_right.PNG",0)
SIFTMATCH(img1,img2)