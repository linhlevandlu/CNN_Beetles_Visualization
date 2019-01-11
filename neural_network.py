try:
	import cPickle as pickle
except ImportError:
	import pickle
import sys
import cv2
import numpy as np
from nolearn.lasagne import NeuralNet
import lasagne

class neuralNetwork:
    def __init__(self,testImage, trained_model):
        #self.cnn = network
        self.pklfile = trained_model
        self.image = testImage

    def predict(self):
        X = self.loadImage()
        net = None
        sys.setrecursionlimit(1000000)
        with open(self.pklfile, 'rb') as f:
            net = pickle.load(f)
        y_pred = net.predict(X)
        #layers = lasagne.layers.get_all_layers(net.layers)
        #output_shapes = lasagne.layers.get_output_shape(layers)
	    #outputs = lasagne.layers.get_output(layers,inputs = X)	
        #print(output_shapes)

        return y_pred

    def loadImage(self):
        img = cv2.imread(self.image, cv2.IMREAD_GRAYSCALE)
        h,w = img.shape[0],img.shape[1]
        print(img.shape)
        img = img.reshape(-1)
        img = img/255.
        img = img.astype(np.float32)
        img = img.reshape(-1,1,h,w)
        print(img.shape)
        return img
        
