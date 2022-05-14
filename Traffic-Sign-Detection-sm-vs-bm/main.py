import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from skimage.feature import blob_dog, blob_log, blob_doh
import imutils
import argparse
import os
import math
import sys
import random as rnd

from classification import training, getLabel, get_hog

SIGNS = ["BATMAN",
         "SPIDERMAN",
         "NEITHER"]

# Clean all previous file
def clean_images():
	file_list = os.listdir('./')
	for file_name in file_list:
		if '.png' in file_name:
			os.remove(file_name)
### Preprocess image
def constrastLimit(image):
    img_hist_equalized = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
#    channels = cv2.split(img_hist_equalized)
#    channels[0] = cv2.equalizeHist(channels[0])
#    img_hist_equalized = cv2.merge(channels)
    y, cr, cb = cv2.split(img_hist_equalized)
    y = cv2.equalizeHist(y)
    img_hist_equalized = cv2.merge((y, cr, cb))
    img_hist_equalized = cv2.cvtColor(img_hist_equalized, cv2.COLOR_YCrCb2BGR)
    return img_hist_equalized

def LaplacianOfGaussian(image):
    LoG_image = cv2.medianBlur(image, 3)       # paramter 
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    LoG_image = cv2.dilate(LoG_image, kernel5)
    LoG_image = cv2.erode(LoG_image, kernel3)

    gray = cv2.cvtColor(LoG_image, cv2.COLOR_BGR2GRAY)
    LoG_image = cv2.Laplacian(gray, cv2.CV_8U, 3, 3, 2)       # parameter
    LoG_image = cv2.convertScaleAbs(LoG_image)
    return LoG_image
    
def binarization(image):
    thresh = cv2.threshold(image,20,255,cv2.THRESH_BINARY)[1]
    #thresh = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 5)
    return thresh

def preprocess_image(image):
    image = constrastLimit(image)
    image = LaplacianOfGaussian(image)
    image = binarization(image)
    return image

"""
# Find Signs
def removeSmallComponents(image, threshold):
    #find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    img2 = np.zeros((output.shape),dtype = np.uint8)
    #for every component in the image, you keep it only if it's above threshold
    for i in range(0, nb_components):
        if sizes[i] >= threshold:
            img2[output == i + 1] = 255
    #cv2.imshow("Connected Comp", img2)
    return img2
"""

def removeSmallComponents(image, threshold):
    #find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    img2 = np.zeros((output.shape),dtype = np.uint8)

    #for every component in the image, you keep it only if it's above threshold
    for i in range(0, nb_components):
        if sizes[i] >= threshold:
            img2[output == i + 1] = 255
    
    mask = np.zeros(img2.shape,dtype="uint8") 
    #cv2.imshow('image',image)
    for i in range(1, nb_components): #filtering out areas for components

        area = stats[i,cv2.CC_STAT_AREA]       
        if area > 10:
            filt_components = (output==i).astype('uint8')
            mask = cv2.bitwise_or(mask,filt_components)

    boxed_conn_comps, cnt = draw_contours(filt_components, mask)
    #cv2.imshow("BoxedConnComps", boxed_conn_comps)
    #cv2.imshow("Connected Comp", img2)

    return img2, cnt

def draw_contours(pix_labels, thresh):
    num_labels = np.max(pix_labels) + 1

    boxed_comps_img = np.zeros([pix_labels.shape[0], pix_labels.shape[1], 3], dtype=np.uint8)
    boxed_comps_img[:,:,:] = 0

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]

    rnd.seed()

    # fill regions with random colors
    for i,_ in enumerate(contours):
        one_pix_hsv = np.zeros([1,1,3],dtype=np.uint8)
        one_pix_hsv[0,0,:] = [ rnd.randint(0,255), rnd.randint(150,255), rnd.randint(200,255) ]
        bgr_color = cv2.cvtColor (one_pix_hsv, cv2.COLOR_HSV2BGR)[0,0].tolist()
        mask = np.zeros(thresh.shape,np.uint8)
        cv2.drawContours(boxed_comps_img,contours,i,bgr_color,-1)

    return boxed_comps_img, contours

def findContour(image):
    #find contours in the thresholded image
    cnts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#    print (imutils.is_cv2())
#    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts = cnts[0]
    return cnts

def contourIsSign(perimeter, centroid, threshold):
    #  perimeter, centroid, threshold
    # # Compute signature of contour
    result=[]
    for p in perimeter:
        p = p[0]
        distance = sqrt((p[0] - centroid[0])**2 + (p[1] - centroid[1])**2)
        result.append(distance)
    max_value = max(result)
    signature = [float(dist) / max_value for dist in result ]
    # Check signature of contour.
    temp = sum((1 - s) for s in signature)
    temp = temp / len(signature)
    if temp < threshold: # is  the sign
        return True, max_value + 2
    else:                 # is not the sign
        return False, max_value + 2

#crop sign 
def cropContour(image, center, max_distance):
    width = image.shape[1]
    height = image.shape[0]
    top = max([int(center[0] - max_distance), 0])
    bottom = min([int(center[0] + max_distance + 1), height-1])
    left = max([int(center[1] - max_distance), 0])
    right = min([int(center[1] + max_distance+1), width-1])
    print(left, right, top, bottom)
    return image[left:right, top:bottom]

def cropSign(image, coordinate):
    width = image.shape[1]
    height = image.shape[0]
    top = max([int(coordinate[0][1]), 0])
    bottom = min([int(coordinate[1][1]), height-1])
    left = max([int(coordinate[0][0]), 0])
    right = min([int(coordinate[1][0]), width-1])
    #print(top,left,bottom,right)
    return image[top:bottom,left:right]

def findLargestSign(image, contours, threshold, distance_theshold):
    max_distance = 0
    coordinate = None
    sign = None
#    print (len(contours))
#    M = [None]*len(contours)
    for c in contours:
#        print (type(c))
#        print (c.shape)
#        print (c.dtype)
        M = cv2.moments(c)
#        M = cv2.moments(np.uint8(c))
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        is_sign, distance = contourIsSign(c, [cX, cY], 1-threshold)
        if is_sign and distance > max_distance and distance > distance_theshold:
            max_distance = distance
            coordinate = np.reshape(c, [-1,2])
            left, top = np.amin(coordinate, axis=0)
            right, bottom = np.amax(coordinate, axis = 0)
            coordinate = [(left-2, top-2), (right+3,bottom+1)]
            sign = cropSign(image,coordinate)
    return sign, coordinate

def findSigns(image, contours, threshold, distance_theshold):
    signs = []
    coordinates = []
    for c in contours:
        # compute the center of the contour
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        is_sign, max_distance = contourIsSign(c, [cX, cY], 1-threshold)
        if is_sign and max_distance > distance_theshold:
            sign = cropContour(image, [cX, cY], max_distance)
            signs.append(sign)
            coordinate = np.reshape(c, [-1,2])
            top, left = np.amin(coordinate, axis=0)
            right, bottom = np.amax(coordinate, axis = 0)
            coordinates.append([(top-2,left-2),(right+1,bottom+1)])
    return signs, coordinates

def localization(image, min_size_components, similitary_contour_with_circle, model):
    
    original_image = image.copy()
    
    binary_image = preprocess_image(image)
    #cv2.imshow('Preprocess', binary_image)
    binary_image, _ = removeSmallComponents(binary_image, min_size_components)
    #cv2.imshow('Rm Small Components', binary_image)
    #binary_image = cv2.bitwise_and(binary_image, binary_image, mask=remove_other_color(image))
    #cv2.imshow('BINARY IMAGE', binary_image)
    #cv2.imshow('Remove color', binary_image)
    #binary_image = remove_line(binary_image)
    #cv2.imshow('Remove line', binary_image)
    
    contours = findContour(binary_image)
    #print(contours)
    #signs, coordinates = findSigns(image, contours, similitary_contour_with_circle, 15)
    sign, coordinate = findLargestSign(original_image, contours, similitary_contour_with_circle, 15)
    
    text = ""
    sign_type = -1
    i = 0

    if sign is not None:
        sign_type = getLabel(model, sign)
        sign_type = sign_type if sign_type <= 8 else 8
        text = SIGNS[sign_type]
        #cv2.imwrite(str(count)+'_'+text+'.png', sign)
     
        cv2.rectangle(original_image, coordinate[0],coordinate[1], (0, 255, 0), 1)
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(original_image, text, (coordinate[0][0], coordinate[0][1] -15), font, 1, (0,0,255), 2, cv2.LINE_4)

    return coordinate, original_image, sign_type, text

def remove_line(img):
    gray = img.copy()
    edges = cv2.Canny(gray, 50, 100, apertureSize = 5)
    #cv2.imshow('Edges', edges)
    minLineLength = 7
    maxLineGap = 5
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 15, minLineLength, maxLineGap)

    mask = np.ones(img.shape[:2], dtype="uint8") * 255
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(mask,(x1,y1),(x2,y2),(0,0,0),2)
    return cv2.bitwise_and(img, img, mask=mask)

def remove_other_color(img):
    frame = cv2.GaussianBlur(img, (3,3), 0) 
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([100,128,0])
    upper_blue = np.array([215,255,255])
    # Threshold the HSV image to get only blue colors
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    lower_white = np.array([0,0,128], dtype=np.uint8)
    upper_white = np.array([255,255,255], dtype=np.uint8)
    # Threshold the HSV image to get only blue colors
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    lower_black = np.array([0,0,0], dtype=np.uint8)
    upper_black = np.array([170,150,50], dtype=np.uint8)

    mask_black = cv2.inRange(hsv, lower_black, upper_black)

    mask_1 = cv2.bitwise_or(mask_blue, mask_white)
    mask = cv2.bitwise_or(mask_1, mask_black)
    # Bitwise-AND mask and original image
    #res = cv2.bitwise_and(frame,frame, mask= mask)
    return mask



class FeatureMatcher():


    def __init__(self, detector, descriptor, matcher):
        
        # check if combination is supported
        self._checkFeatureCombo(detector, descriptor, matcher)

        self.detector = detector.upper()
        self.descriptor = descriptor.upper()
        self.matcher = matcher.upper()


    def _checkFeatureCombo(self, detector, descriptor, matcher):

        # support following combinations:
        #   FAST - BRIEF - FLANN
        #   ORB - ORB - BF
        #   SIFT - SIFT - FLANN

        # check that given combination is supported
        combo = detector.upper() + ' - ' + descriptor.upper()  + ' - ' + matcher.upper()
        supported_combos = ['FAST - BRIEF - FLANN', 
                            'ORB - ORB - BF', 
                            'SIFT - SIFT - FLANN']
        if combo not in supported_combos:
            print(f'\nFeatureMatcher: given combination {combo} not supported')
            print('The following combinations are supported: ')
            for x in supported_combos:
                print(f'\t{x}')
            print('\n')
            exit()


    def _crossCheck(self, matches, matches_rev, twoLevel=False):
        good_matches = []

        for i in range(len(matches)):
            if isinstance(matches[i], (tuple)):
                m, n = matches[i]
            else:
                m = matches[i]

            for j in range(len(matches_rev)):
                if isinstance(matches_rev[j], (tuple)):
                    p, q = matches_rev[j]
                else:
                    p = matches_rev[j]

                if m.queryIdx == p.trainIdx and p.queryIdx == m.trainIdx:
                    good_matches.append (m)
                elif twoLevel:
                    if m.queryIdx == q.trainIdx and q.queryIdx == m.trainIdx:
                        good_matches.append (m)
                    elif n.queryIdx == p.trainIdx and p.queryIdx == n.trainIdx:
                        good_matches.append (n)
                    elif n.queryIdx == q.trainIdx and q.queryIdx == n.trainIdx:
                        good_matches.append (n)

        return good_matches


    def _lowesRatio(self, matches, ratio = .7):
        # use lowes ratio test to filter for good matches
        good_matches = []
        for m, n in matches:
            if m.distance < ratio * n.distance:
                good_matches.append(m)
        return good_matches


    def detect(self, img):

        detectors = {'FAST': cv2.FastFeatureDetector_create(), 
                     'ORB': cv2.ORB_create(), 
                     'SIFT': cv2.SIFT_create()}

        det = detectors[self.detector]
        return det.detect(img, None)

    
    def compute(self, img, kp):

        descriptors = {'BRIEF': cv2.xfeatures2d.BriefDescriptorExtractor_create(), 
                       'ORB': cv2.ORB_create(), 
                       'SIFT': cv2.SIFT_create()}

        desc = descriptors[self.descriptor]
        return desc.compute(img, kp)


    def detectAndCompute(self, img):

        detectors_with_descriptiors = {'ORB': cv2.ORB_create(), 
                                       'SIFT': cv2.SIFT_create()}
        try:
            det_and_desc = detectors_with_descriptiors[self.detector]
        except:
            print("detectAndCompute: given detector cannot act as descriptor")
            exit()
        return det_and_desc.detectAndCompute(img, None)


    def match(self, des_query, des_train, FLANN_INDEX_KDTREE = 1, trees = 5, checks = 50, crossCheck = False, k = 2, lowes = .7, filter = 0):

        # convert to float 32
        des_query = np.float32(des_query)
        des_train = np.float32(des_train)  

        matchers = {'FLANN': cv2.FlannBasedMatcher(dict(algorithm = FLANN_INDEX_KDTREE, trees = trees), 
                                                   dict(checks = checks)),
                    'BF': cv2.BFMatcher(cv2.NORM_L2SQR, crossCheck = crossCheck)}
        mat = matchers[self.matcher]
        matches = mat.knnMatch(des_query, des_train, k = k)

        # use given match filer
        matches_rev = mat.knnMatch(des_train, des_query, k = k)
        match_filters = [matches,                                                  # no filter
                         self._lowesRatio(matches, lowes),                         # lowes ratio test
                         self._crossCheck(matches, matches_rev, twoLevel = False), # crossCheck, twoLevel = False
                         self._crossCheck(matches, matches_rev, twoLevel = True)]  # crossCheck, twoLevel = True

        # return good_matches
        return match_filters[filter]


    def drawMatches(self, img_query, good_matches, kp_query):

        if isinstance(good_matches, (tuple)):
            good_matches = [m for m, n in good_matches]

        """
        # only consider if number of good matches meets min
        if len(good_matches) < MIN_NUM_GOOD_MATCHES:
            print("Not enough matches good were found - %d/%d" % (len(good_matches), MIN_NUM_GOOD_MATCHES))
            # return no matches image
            no_matches = cv2.imread('no_matches.jpg')
            return no_matches
        """

        # matching points in query image
        src_pts = np.float32(
            [kp_query[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        """
        # matching points in training image
        dst_pts = np.float32(
            [kp_train[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # homography matrix
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        h, w, _ = img_query.shape
        src_corners = np.float32(
            [[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        dst_corners = cv2.perspectiveTransform(src_corners, M)
        dst_corners = dst_corners.astype(np.int32)

        # Draw the bounds of the matched region based on the homography.
        
        img_train_copy = img_train.copy()
        num_corners = len(dst_corners)
        for i in range(num_corners):
            x0, y0 = dst_corners[i][0]
            if i == num_corners - 1:
                next_i = 0
            else:
                next_i = i + 1
            x1, y1 = dst_corners[next_i][0]
            cv2.line(img_train_copy, (x0, y0), (x1, y1), 255, 3, cv2.LINE_AA)

        # Draw the matches that passed the ratio test.
        #img_matches = cv2.drawMatches(
        #    img_query, kp_query, img_train_copy, kp_train, good_matches, None,
        #    matchColor = (0, 255, 0), singlePointColor = None,
        #    flags=2)
        """

        for pt in src_pts:
            img_query = cv2.circle(img_query, (pt[0], pt[1]), radius=0, color=(0,255,0), thickness=-1)

        return img_query

    def _matchBestTrainImg(self, img_query, test_img_loc = './dataset/0', FLANN_INDEX_KDTREE = 1, trees = 5, checks = 50, crossCheck = False, k = 2, lowes = .7, filter = 0):

        kp_query = self.detect(img_query)
        kp_query, des_query = self.compute(img_query, kp_query)

        # get all images (files ending with png or jpg) in training image folder
        training_imgs = [x for x in os.listdir(test_img_loc) if x.endswith('.png') or x.endswith('.jpg')]

        train_kp_des = []
        # compute kp / des for each training image and store in list
        for img_name in training_imgs:
            # read in image file
            img_path = f'{test_img_loc}\{img_name}'
            train_img = cv2.imread(img_path)
            # compute kp and des and add to train_kp_des list
            kp_train = self.detect(train_img)
            kp_train, des_train = self.compute(train_img, kp_train)
            # print(f'{img_name} kp: {len(kp_train)}, des: {len(des_train)}')
            train_kp_des.append([kp_train, des_train])


        matches = []
        # iterate over all traing kp / des pairs
        for index, train_det in enumerate(train_kp_des):
            if train_det[0] is None or train_det[1] is None:
                continue
            # find matches against query image
            good_matches = self.match(des_query, train_det[1], FLANN_INDEX_KDTREE, trees, checks, crossCheck, k, lowes, filter)
            print(f'{training_imgs[index]} has {len(good_matches)} matches with query image')
            # add to matches
            matches.append(matches)

        matched = self.drawMatches(img_query, 
                                    matches, 
                                    kp_query)
        
        return matched


def resize_img(img, width = 250):
    r = width / img.shape[1]
    dim = (width, int(img.shape[0] * r))
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

def matchDescriptors(img, training_imgs):

    sift = cv2.SIFT_create()
    matcher = cv2.FlannBasedMatcher(dict(algorithm = 0, trees = 15), 
                                     dict(checks = 200))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_kp, img_desc = sift.detectAndCompute(gray, None)

    min_good_matches = 0
    kp_good_matches = []
    src_points = []
    dst_points = []
    for i, train in enumerate(training_imgs):
        # print("Matching training image: ", i + 1, " out of ", len(training_imgs))
        train_gray = cv2.cvtColor(train, cv2.COLOR_BGR2GRAY)

        train_kp, train_desc = sift.detectAndCompute(train_gray, None)
        # match img descriptors to each training image descriptor
        matches = matcher.knnMatch(img_desc, train_desc, k = 2)
        # filter matches with lowes ratio
        good_matches = []
        for m, n in matches:
            if m.distance < .7 * n.distance:
                good_matches.append(m)

        if len(good_matches) >= min_good_matches:
            good_kps = [img_kp[m.queryIdx] for m in good_matches]
            kp_good_matches = kp_good_matches + good_kps

            src_pts = [img_kp[m.queryIdx].pt for m in good_matches]
            dst_pts = [train_kp[m.trainIdx].pt for m in good_matches]

            src_points = src_points + src_pts
            dst_points = dst_points + dst_pts
    
    src_points = np.float32(src_points).reshape(-1,1,2)
    dst_points = np.float32(dst_points).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

    pts = src_points[mask==1]
    min_x, min_y = np.int32(pts.min(axis=0))
    max_x, max_y = np.int32(pts.max(axis=0))
    match_img = cv2.rectangle(img, (min_x, min_y), (max_x,max_y), 255,2)
    #match_img = cv2.drawKeypoints(match_img, kp_good_matches, match_img, (255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cropped_img = img[min_y:max_y, min_x:max_x]
    # return good_matches
    # return match_filters[filter]
    return match_img, cropped_img


def main():
	# clean previous image    
    clean_images()
    # training phase
    model = training()

    # read in test images from command line instead of video
    test_imgs = []
    test_img_files = []
    for img_path in os.listdir("./dataset/test"):
        if any(x in img_path for x in ['.jpg', '.jpeg', '.png']):
            img = cv2.imread(f"./dataset/test/{img_path}")
            test_imgs.append(img)
            test_img_files.append(img_path)

    
    training_imgs = []
    for img_path in os.listdir("./dataset/0"):
        if any(x in img_path for x in ['.jpg', '.jpeg', '.png']):
            img = cv2.imread(f"./dataset/0/{img_path}")
            img = resize_img(img)
            training_imgs.append(img)
    for img_path in os.listdir("./dataset/1"):
        if any(x in img_path for x in ['.jpg', '.jpeg', '.png']):
            img = cv2.imread(f"./dataset/1/{img_path}")
            img = resize_img(img)
            training_imgs.append(img)
    for img_path in os.listdir("./dataset/2"):
        if any(x in img_path for x in ['.jpg', '.jpeg', '.png']):
            img = cv2.imread(f"./dataset/2/{img_path}")
            img = resize_img(img)
            training_imgs.append(img)
    # randomly choose n images
    # training_imgs = rnd.sample(training_imgs, 10)

    for i, test_img in enumerate(test_imgs):

        #frame = cv2.resize(frame, (640,int(height/(width/640))))
        test_img = resize_img(test_img)

        #image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        #coordinate, image, sign_type, text = localization(test_img, 300, 0.1, model)

        result, cropped = matchDescriptors(test_img, training_imgs)
        #cv2.imshow("Result", cropped)
        
        sign_type = getLabel(model, cropped)
        cv2.imshow(f"Image: {test_img_files[i]}, Prediction: {SIGNS[sign_type]}", result)
        print(f"\nImage: {test_img_files[i]}")
        print("Sign: {}".format(sign_type))
        print(f"Prediction: {SIGNS[sign_type]}")

        """
        print(coordinate)
        if coordinate is not None:
            # cv2.rectangle(image, coordinate[0], coordinate[1], (255, 255, 255), 1)
            cv2.rectangle(image, coordinate[0],coordinate[1], (0, 255, 0), 1)
            font = cv2.FONT_HERSHEY_PLAIN
            cv2.putText(image,text,(coordinate[0][0], coordinate[0][1] -15), font, 1,(0,0,255),2,cv2.LINE_4)
        


        cv2.imshow(f'Result: {test_img_files[i]} Prediction: {SIGNS[sign_type]}', image)
        break
        """

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 


if __name__ == '__main__':

    main()

    # problem with edge detection, not finding entire closed cover

