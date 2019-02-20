import numpy as np
#from sklearn.preprocessing import normalize
import cv2

print('loading images...')

imgL = cv2.imread('D:\\tsucuba_left.PNG')
imgR = cv2.imread('D:\\tsucuba_right.PNG')


# SGBM Parameters -----------------
window_size = 3                    

left_matcher = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=64,             
    blockSize=5,
    P1=8 * 3 * window_size ** 2,    
    P2=32 * 3 * window_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=15,
    speckleWindowSize=0,
    speckleRange=2,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)
right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
# FILTER Parameters
lmbda = 80000
sigma = 1.2
visual_multiplier = 1.0
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)
print('computing disparity...')
displ = left_matcher.compute(imgL, imgR)  
dispr = right_matcher.compute(imgR, imgL)  
displ = np.int16(displ)
dispr = np.int16(dispr)
filteredImg = wls_filter.filter(displ, imgL, None, dispr)  

filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
filteredImg = np.uint8(filteredImg)
cv2.imwrite('D://task2_disparity.jpg',filteredImg)