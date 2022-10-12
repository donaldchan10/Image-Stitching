import numpy as np
import cv2 as cv

def display(img):
    """
    Function used to display a given image in an external window
    
    Function takes 1 parameter:

    img: The image that you want to display
    """
    # Display the image in a window
    cv.imshow('image', img)

    # Stops window from closing
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()

def stitch(right_img, left_img, min_matches=10):
    """
    Function takes in 3 parameters:

    right_img: Image that you want on the right side

    left_img: Image that you want on the left side
    
    min_matches: The number of minimum keypoint matches between the two images. Default value is 10.
    
    Note: min_matches is used so that two images that have less matches than min_matches will be considered as "too different"
    and will be un-stitchable.

    Function used to stitch together 2 similar images.

    First, it detects features in both images. This is done using OpenCV's SIFT class, 
    which detects keypoints and creates descriptors that represent those keypoints

    Next, Those descriptors are fed into a FLANN based matcher. 
    The matcher determines if two keypoints are similar enough to be considered a match. 
    This is provided by OpenCV's FLANNBasedMatcher class.

    The resulting matches are than filtered again using Lowe's Ratio Test.
    Matches that pass Lowe's Ratio Test are considered to be "good" matches.

    The good matches will be used to determine homography matrix.
    By passing the good matches to OpenCV's findHomography(), a corresponding
    homography matrix is returned.

    The homography matrix is then applied to the right image using OpenCV's
    warpPerspective()

    Finally, the original left image and the warped right image are placed onto the same
    canvas, resulting in a stitched image.
    """
    # Read in the images
    right_image = cv.imread(right_img)
    left_image = cv.imread(left_img)

    # Copy and convert the original images into grayscale for feature detection
    img1 = cv.cvtColor(right_image,cv.COLOR_BGR2GRAY)
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
    if len(matches[:,0]) >= min_matches:
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
