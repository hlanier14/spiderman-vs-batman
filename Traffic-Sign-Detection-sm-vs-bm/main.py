import cv2
import numpy as np
import os
import sys
import itertools

from classification import HOGBOWSVMPipeline


def resizeImage(img, width = 250):
    r = width / img.shape[1]
    dim = (width, int(img.shape[0] * r))
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def getImages(folder, width = 250):
    imgs = []
    img_names = []
    for img_path in os.listdir(folder):
        if any(x in img_path for x in ['.jpg', '.jpeg', '.png']):
            img = cv2.imread(f"{folder}/{img_path}")
            img = resizeImage(img, width)
            imgs.append(img)
            img_names.append(img_path)
    return imgs, img_names


class ObjectDetector:

    # find comic book cover in test image
    # detect - extract - match
    # SIFT - SIFT - FLANN
    # match descriptors between the test image and all training_imgs 

    def __init__(self, matcher_algorithm = 1, matcher_trees = 10, matcher_checks = 150, min_good_matches = 0, lowes = .7):
        self.detector = cv2.SIFT_create()
        self.matcher = cv2.FlannBasedMatcher(dict(algorithm = matcher_algorithm, trees = matcher_trees), 
                                             dict(checks = matcher_checks))
        self.min_good_matches = min_good_matches
        self.lowes = lowes

    def detectAndCompute(self, img):
        return self.detector.detectAndCompute(img, None)

    def _getGoodMatches(self, matches):
        good_matches = []
        for m, n in matches:
            if m.distance < self.lowes * n.distance:
                good_matches.append(m)
        return good_matches

    def _drawBoundingBox(self, img, src_points, dst_points):

        img_copy = img.copy()

        M, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

        pts = src_points[mask==1]
        min_x, min_y = np.int32(pts.min(axis=0))
        max_x, max_y = np.int32(pts.max(axis=0))
        match_img = cv2.rectangle(img_copy, (min_x, min_y), (max_x,max_y), 255, 2)
        #match_img = cv2.drawKeypoints(match_img, kp_good_matches, match_img, (255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # return image with bounding both and cropped image
        return match_img, img[min_y:max_y, min_x:max_x]

    def match(self, img, training_imgs):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_kp, img_desc = self.detectAndCompute(gray)

        kp_good_matches = []
        src_points = []
        dst_points = []
        for train in training_imgs:
            # print("Matching training image: ", i + 1, " out of ", len(training_imgs))
            train_gray = cv2.cvtColor(train, cv2.COLOR_BGR2GRAY)

            train_kp, train_desc = self.detectAndCompute(train_gray)
            # match img descriptors to each training image descriptor
            matches = self.matcher.knnMatch(img_desc, train_desc, k = 2)
            # filter matches with lowes ratio
            good_matches = self._getGoodMatches(matches)

            # keep track of all good match points between img and all training imgs
            if len(good_matches) >= self.min_good_matches:
                kp_good_matches = kp_good_matches + [img_kp[m.queryIdx] for m in good_matches]
                src_points = src_points + [img_kp[m.queryIdx].pt for m in good_matches]
                dst_points = dst_points + [train_kp[m.trainIdx].pt for m in good_matches]
        
        src_points = np.float32(src_points).reshape(-1,1,2)
        dst_points = np.float32(dst_points).reshape(-1,1,2)

        # return img with bounding box and cropped img of pixels in bounding box
        return self._drawBoundingBox(img, src_points, dst_points)


def main():

    COMICS = ["BATMAN",
              "SPIDERMAN",
              "NEITHER"]

    # read in test images and training images
    test_imgs, test_img_files = getImages("./dataset/test")
    #test = 'spiderman4.jpeg'
    #test_imgs = [resizeImage(cv2.imread(f"./dataset/test/{test}"))]
    #test_img_files = [test]
    training_imgs_orig = getImages("./dataset/0")[0] + getImages("./dataset/1")[0] + getImages("./dataset/2")[0]


    # specify parameters for DescriptorMatcher
    matcher_params = {"matcher_algorithm": 0,
                      "matcher_trees": 10,
                      "matcher_checks": 150,
                      "min_good_matches": 0,
                      "lowes": .7}

    matcher = ObjectDetector(**matcher_params)

    model = HOGBOWSVMPipeline(bow_k = 55)
    model.train()

    # find and predict label for comic book cover in each test img
    predictions = []
    for i, test_img in enumerate(test_imgs):
        # get img with bounded box and pixels captured in the box
        result, cropped = matcher.match(test_img, training_imgs_orig)
        #cropped = resizeImage(cropped, width = 700)
        # predict label for bounded pixels
        comic_type = model.predict(cropped)
        # show image with bounded box
        #cv2.imshow(f"Image: {test_img_files[i]} Cropped", cropped)
        cv2.imshow(f"Image: {test_img_files[i]}, Prediction: {COMICS[comic_type]}", result)

        predictions.append(COMICS[comic_type])
    
    true_classes = ["BATMAN", "BATMAN", "BATMAN", "BATMAN",
                    "SPIDERMAN", "SPIDERMAN", "SPIDERMAN", "SPIDERMAN", 
                    "NEITHER", "NEITHER", "NEITHER", "NEITHER"]

    print(predictions)
    print(true_classes)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()



