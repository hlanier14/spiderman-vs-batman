import cv2
import os
import numpy as np


def checkForDuplicates():

    # load in all images from training_imgs and test_imgs folders
    training_imgs = [f'training_imgs/{x}' for x in os.listdir('training_imgs') if x.endswith('.png') or x.endswith('.jpg') or x.endswith('.jpeg') or x.endswith('.jfif')]
    test_imgs = [f'test_imgs/{x}' for x in os.listdir('test_imgs') if x.endswith('.png') or x.endswith('.jpg') or x.endswith('.jpeg') or x.endswith('.jfif')]
    all_imgs = training_imgs + test_imgs

    counter = 1
    # iterate over all images
    for x in all_imgs:
        print(f'Checking {x}... {counter}/{len(all_imgs)}')
        x_img = cv2.imread(x)
        counter += 1
        # iterate over all other images
        for y in all_imgs:
            if x == y:
                continue
            y_img = cv2.imread(y)
            # print match if imgs have same shape and no values are the same
            if x_img.shape == y_img.shape and not(np.bitwise_xor(x_img,y_img).any()):
                print(f'Match: {x} and {y}')
    return True


print(checkForDuplicates())
