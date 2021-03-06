

import os
import cv2
import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle

FTRAIN = ''
FTEST = ''
X_DIV = 128
Y_DIV = 96

def readImage(arrayImgs):
    X = arrayImgs.values
    for i in range(len(X)):
        image = cv2.imread(X[i],cv2.IMREAD_GRAYSCALE)
        image = image.reshape(-1)
        image = image/255.
        image = image.astype(np.float32)
        X[i] = image
    X = np.vstack(X)
    return X

def scaleTarget(target):
	print('Normalize target...')
	evencol = (target[:,::2] - 128)/128
	oddcol = (target[:,1::2] - 96)/96
	rs = np.empty((evencol.shape[0],evencol.shape[1] + oddcol.shape[1]))
	rs[:,::2] = evencol
	rs[:,1::2] = oddcol
	
	return rs

def loaddata(fname = None,test=False):
	if fname == None:
		fname = FTEST if test else FTRAIN
	df = read_csv(os.path.expanduser(fname))
	df = df.dropna()
	imagePath = df['Image']
	X = readImage(imagePath)
	if not test:
		y = df[df.columns[:-1]].values
		y = y.astype(np.float32)
		y = scaleTarget(y)
		X,y = shuffle(X,y,random_state=42)
		y = y.astype(np.float32)
		#print(y)
	else:
		y = None
	return X,y

def revertPredict(y_preds):
    evencol = (y_preds[:,::2] * X_DIV) + X_DIV
    oddcol = (y_preds[:,1::2] * Y_DIV) + Y_DIV
    results = np.empty((evencol.shape[0],evencol.shape[1] + oddcol.shape[1]))
    results[:,::2] = evencol
    results[:,1::2] = oddcol
    return results


# test loaddata method
#X,y = loaddata()
#print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
#    X.shape, X.min(), X.max()))
#print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
#    y.shape, y.min(), y.max()))

# reshape (convert) the data from 49152 to 192x256 (h x w)

def load2d(fname=None,test=False):
    print(fname)
    X,y = loaddata(fname,test=test)
    X = X.reshape(-1,1,256,192)
    if not test:
        print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(X.shape, X.min(), X.max()))
        print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(y.shape, y.min(), y.max()))
    return X,y

