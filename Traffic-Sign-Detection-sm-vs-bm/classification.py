import cv2
import numpy as np
from matplotlib import pyplot as plt
from os import listdir
# local modules
from common import clock, mosaic

#Parameter
SIZE = 64
CLASS_NUMBER = 3


def load_traffic_dataset():
    dataset = []
    labels = []
    for sign_type in range(CLASS_NUMBER):
        sign_list = listdir("./dataset/{}".format(sign_type))
        for sign_file in sign_list:
            if any(x in sign_file for x in ['.jpg', '.jpeg', '.png', '']):
                path = "./dataset/{}/{}".format(sign_type,sign_file)
                print(path)
                img = cv2.imread(path,0)
                img = cv2.resize(img, (SIZE, SIZE))
                img = np.reshape(img, [SIZE, SIZE])
                dataset.append(img)
                labels.append(sign_type)
    return np.array(dataset), np.array(labels)


def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SIZE*skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SIZE, SIZE), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

class StatModel(object):
    def load(self, fn):
        self.model.load(fn)  # Known bug: https://github.com/opencv/opencv/issues/4969
    def save(self, fn):
        self.model.save(fn)

class SVM(StatModel):
    def __init__(self, C = 12.5, gamma = 0.50625):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        return self.model.predict(samples)[1].ravel()


def evaluate_model(model, data, samples, labels):
    resp = model.predict(samples)
    print(resp)
    err = (labels != resp).mean()
    print('Accuracy: %.2f %%' % ((1 - err)*100))

    confusion = np.zeros((10, 10), np.int32)
    for i, j in zip(labels, resp):
        confusion[int(i), int(j)] += 1
    print('confusion matrix:')
    print(confusion)

    vis = []
    for img, flag in zip(data, resp == labels):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if not flag:
            img[...,:2] = 0
        
        vis.append(img)
    return mosaic(16, vis)

def preprocess_simple(data):
    return np.float32(data).reshape(-1, SIZE*SIZE) / 255.0


def get_hog() : 
    winSize = (20,20)
    blockSize = (10,10)
    blockStride = (5,5)
    cellSize = (10,10)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradient = True

    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradient)

    return hog
    affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR


class BOW:

    def __init__(self):
        # k = 50 gives same output as original code
        self.k = 45
        self.vocabulary = None

    def cluster(self, descriptors):
        # cluster descriptors using kmeans cv2 function
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_RANDOM_CENTERS
        # vocabulary is array of centroid points
        _, _, self.vocabulary = cv2.kmeans(descriptors, self.k, None, criteria, 10, flags)

    def generate_histogram(self, descriptors):

        # reshape descriptors if dim does not match with vocabulary
        if descriptors.ndim == 1:
            descriptors = np.array([descriptors])

        # k-length array to store distances of each vocabulary element in the given descriptors
        histogram = np.zeros((len(self.vocabulary), 1))

        # histogram of distances to each centroid
        for i, word in enumerate(self.vocabulary):
            # get euclidian distance
            dist = np.linalg.norm(descriptors - word)
            histogram[i] = dist

        # flatten array into a row
        return histogram.flatten()

bow = BOW()

def training():
    print('Loading data from data.png ... ')
    # Load data.
    #data, labels = load_data('data.png')
    data, labels = load_traffic_dataset()
    print('Shuffle data ... ')
    # Shuffle data
    rand = np.random.RandomState(10)
    shuffle = rand.permutation(len(data))
    data, labels = data[shuffle], labels[shuffle]
    
    print('Deskew images ... ')
    data_deskewed = list(map(deskew, data))
    
    print('Defining HoG parameters ...')
    # HoG feature descriptor
    hog = get_hog()

    print('Calculating HoG descriptor for every image ... ')
    hog_descriptors = []
    # get hog descriptors
    for img in data_deskewed:
        descriptors = hog.compute(img)
        hog_descriptors.append(descriptors)
    hog_descriptors = np.squeeze(hog_descriptors)

    # cluster descriptors using BOW
    bow.cluster(hog_descriptors)

    # generating histograms for each descriptor
    # each image should have a k length vector of # of times each vocab element shows up in the image
    histograms = np.empty((len(hog_descriptors), bow.k))
    for i, desc in enumerate(hog_descriptors):
        hist = bow.generate_histogram(desc)
        histograms[i] = hist

    # use all training images
    # print('Spliting data into training (90%) and test set (10%)... ')
    # train_n = int(0.9 * len(data))
    # data_hist_train, data_hist_test = np.split(np.float32(histograms), [train_n])
    # labels_train, labels_test = np.split(labels, [train_n])
    
    print('Training SVM model ...')
    model = SVM()
    model.train(np.float32(histograms), labels)

    print('Saving SVM model ...')
    model.save('data_svm.dat')
    return model


def getLabel(model, data):
    gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    img = [cv2.resize(gray,(SIZE,SIZE))]
    #print(np.array(img).shape)
    img_deskewed = list(map(deskew, img))
    hog = get_hog()
    hog_descriptors = np.array([hog.compute(img_deskewed[0])])
    hog_descriptors = np.reshape(hog_descriptors, [-1, hog_descriptors.shape[1]])
    hog_descriptors = np.float32(hog_descriptors)

    # generate histogram for test image
    hist = bow.generate_histogram(hog_descriptors)
    hist = np.float32(hist)

    # pass reshaped histogram into SVM model
    # test image histogram should match training image histograms
    pred = model.predict(np.array([hist]))

    # predicted value should be an integer
    return int(pred[0])

