import numpy as np
import cv2 as cv

def display(img):
    # Display the image in a window
    cv.imshow('image', img)

    # Stops window from closing
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()

def stitch(right_img, left_img):
    right_image = cv.imread(right_img)
    img1 = cv.cvtColor(right_image,cv.COLOR_BGR2GRAY)
    left_image = cv.imread(left_img)
    img2 = cv.cvtColor(left_image,cv.COLOR_BGR2GRAY)

    # SIFT Brute Force
    sift = cv.SIFT.create()

    # Feature Detection using SIFT Descriptors
    kp1, desc1 = sift.detectAndCompute(img1, None)
    kp2, desc2 = sift.detectAndCompute(img2, None)

    # FLANN Based Matcher for matching descriptors
    # First define dict index_params and search_params for FLANN Matcher
    index_params = dict(algorithm = 0, trees = 5)
    search_params = dict()

    # Call FLANN Matcher
    flann = cv.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(desc1,desc2,k=2)

    # Apply David Lowe's Ratio Test to obtain "good" matches
    good = []
    for match in matches:
        if match[0].distance < 0.7*match[1].distance:
            good.append(match)
    matches = np.asarray(good)

    # Create homograhy matrix if there are enough matches (more than 10)
    # Raise an Error if there are not enough matches
    if len(matches[:,0]) >= 10:
        src = np.float32([ kp1[match.queryIdx].pt for match in matches[:,0] ]).reshape(-1,1,2)
        dst = np.float32([ kp2[match.trainIdx].pt for match in matches[:,0] ]).reshape(-1,1,2)
        H, masked = cv.findHomography(src, dst, cv.RANSAC, 5.0)
    else:
        raise AssertionError("Cannot find enough keypoints.")

    # Apply the homography matrix to the right image
    dst = cv.warpPerspective(right_image,H,(right_image.shape[1] + right_image.shape[1], left_image.shape[0]))

    # Stitch the original left image and the warped right image together
    dst[0:left_image.shape[0], 0:left_image.shape[1]] = left_image

    # Display the final stitched image
    display(dst)
