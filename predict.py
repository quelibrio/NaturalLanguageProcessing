import keras
from keras.models import Model
from keras import backend as K
from gensim.models import Word2Vec
import W2VProcessing as processing
import pytextvec as pytextvec

model = keras.models.load_model('data/model.h5')

print("Loading Gensim Model...")
word_vectors = Word2Vec.load('data/word2vec/500features_10minwords_10context')
maxlen = 10
X = processing.loadXTrain('/home/quelibrio/Work/Bevrage/BevBox/receiptcomprehension/data/SamplePrintScans/bevmo_sample_receipt_01.txt')
x_test = processing.comments2Matrix(X, word_vectors,maxlen)
print(x_test)
prediction = model.predict(x=x_test, batch_size=1000)

print(prediction)

#fpr_keras, tpr_keras, thresholds_keras = roc_curve(ytest.argmax(),prediction.argmax())

#from sklearn.metrics import auc
#auc_keras = auc(fpr_keras, tpr_keras)

#import tensorflow as tf
#prediction1=tf.argmax(logits,1)
#print(prediction[0])
#print(prediction)

import numpy as np
np.savetxt('data/predictions.csv', prediction, delimiter=',')