import cv2
import numpy as np
from os import listdir


class BOW:

    def __init__(self, k = 45):
        # k = 50 gives same output as original code
        self.k = k
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


class SVM:

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


class HOG:
    
    def __init__(self):
        self.hog = cv2.HOGDescriptor(_winSize = (20,20), 
                                    _blockSize = (10,10), 
                                    _blockStride = (5,5), 
                                    _cellSize = (10,10), 
                                    _nbins = 9, 
                                    _derivAperture = 1, 
                                    _winSigma = -1., 
                                    _histogramNormType = 0, 
                                    _L2HysThreshold = 0.2, 
                                    _gammaCorrection = 1,
                                    _nlevels = 64, 
                                    _signedGradient = True)

    def compute(self, img):
        return self.hog.compute(img)


class HOGBOWSVMPipeline:


    def __init__(self, bow_k = 45, svm_c = 12.5, svm_gamma = 0.50625, img_size = 64, class_number = 3):
        self.hog = HOG()
        self.bow = BOW(bow_k)
        self.svm = SVM(C = svm_c, gamma = svm_gamma)
        self.SIZE = img_size
        self.CLASS_NUMBER = class_number


    def _loadTrainingDataset(self, base_folder = './dataset'):
        dataset = []
        labels = []
        for sign_type in range(self.CLASS_NUMBER):
            sign_list = listdir(f"{base_folder}/{sign_type}")
            for sign_file in sign_list:
                if any(x in sign_file for x in ['.jpg', '.jpeg', '.png']):
                    path = f"{base_folder}/{sign_type}/{sign_file}"
                    # print(path)
                    img = cv2.imread(path,0)
                    img = cv2.resize(img, (self.SIZE, self.SIZE))
                    img = np.reshape(img, [self.SIZE, self.SIZE])
                    dataset.append(img)
                    labels.append(sign_type)
        return np.array(dataset), np.array(labels)


    def _deskew(self, img):
        m = cv2.moments(img)
        if abs(m['mu02']) < 1e-2:
            return img.copy()
        skew = m['mu11']/m['mu02']
        M = np.float32([[1, skew, -0.5 * self.SIZE * skew], [0, 1, 0]])
        img = cv2.warpAffine(img, M, (self.SIZE, self.SIZE), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
        return img


    def train(self):
        
        print('Loading training data ... ')
        # Load data.
        data, labels = self._loadTrainingDataset()

        print('Shuffle data ... ')
        # Shuffle data
        rand = np.random.RandomState(10)
        shuffle = rand.permutation(len(data))
        data, labels = data[shuffle], labels[shuffle]
        
        print('Deskew images ... ')
        data_deskewed = list(map(self._deskew, data))
        
        print('Calculating HoG descriptor for every image ... ')
        hog_descriptors = []
        # get hog descriptors
        for img in data_deskewed:
            descriptors = self.hog.compute(img)
            hog_descriptors.append(descriptors)
        hog_descriptors = np.squeeze(hog_descriptors)

        # cluster descriptors using BOW
        self.bow.cluster(hog_descriptors)

        # generating histograms for each descriptor
        # each image should have a k length vector of # of times each vocab element shows up in the image
        histograms = np.empty((len(hog_descriptors), self.bow.k))
        for i, desc in enumerate(hog_descriptors):
            hist = self.bow.generate_histogram(desc)
            histograms[i] = hist
        
        print('Training SVM model ...')
        self.svm.train(np.float32(histograms), labels)

        print('Completed training')


    def predict(self, img):

        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #img = [cv2.resize(gray, (self.SIZE, self.SIZE))]

        img_deskewed = list(map(self._deskew, img))

        hog_descriptors = np.array([self.hog.compute(img_deskewed[0])])
        hog_descriptors = np.reshape(hog_descriptors, [-1, hog_descriptors.shape[1]])
        hog_descriptors = np.float32(hog_descriptors)

        # generate histogram for test image
        hist = self.bow.generate_histogram(hog_descriptors)
        hist = np.float32(hist)

        # pass reshaped histogram into SVM model
        # test image histogram should match training image histograms
        pred = self.svm.predict(np.array([hist]))

        # predicted value should be an integer
        return int(pred[0])

