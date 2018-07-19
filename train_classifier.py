from gensim.models import Word2Vec
import gensim
import numpy as np
import pandas as pd
import W2VProcessing as processing
import pytextvec as pytextvec

#gensim_file = 'data/word2vec/glove.42B.300d.txt'

print("Loading Gensim Model...")
word_vectors = Word2Vec.load('data/word2vec/500features_10minwords_10context')

vocab = word_vectors.wv.vocab
vocab_size = len(vocab)
print(vocab_size)
#print(word_vectors.wv.)
embedding = np.asarray(vocab)
print(embedding)
embed_size = 500 #Embed Size Of Model
maxlen = 10 #Max number of words to use for a specific comment //
max_features = len(word_vectors.wv.vocab) # how many unique words to use (i.e num rows in embedding vector)

print(vocab["relic"])

X = processing.loadXTrain('data/receipt_lines.csv')
y = pd.read_csv('data/receipt_labels.csv', header = None).values # Read Subdrivers

from keras.utils import to_categorical
y_binary = to_categorical(y)

print(y.shape)

train_articles,test_articles,ytrain,ytest = processing.trainTestSplit(X,y_binary,0.2)

xtrain = processing.comments2Matrix(train_articles,word_vectors,maxlen)
xtest = processing.comments2Matrix(test_articles,word_vectors,maxlen)
#rint("xtrain" +xtrain[0])
#print(xtest[0])

embedding_matrix = processing.createEmbeddingMatrix(word_vectors,embed_size)
#print("embeding matrix" + embedding_matrix[0])
import matplotlib

import matplotlib.pyplot as plt

def plot_history(history):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    
    if len(loss_list) == 0:
        print('Loss is missing in history')
        return 
    
    ## As loss always exists
    epochs = range(1,len(history.history[loss_list[0]]) + 1)
    
    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    ## Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_acc_list:    
        plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('metrics.png')
    plt.show()
#plot_history(history)
import keras
class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_acc'))
        self.i += 1
        
        #clear_output(wait=True)
        plt.plot(self.x, self.losses, label="mse")
        plt.plot(self.x, self.val_losses, label="val_mse")
        plt.legend()
        plt.show()
        
plot_losses = PlotLosses()

subdriver_classes = ['order_item', 'other']
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, MaxPooling2D, Reshape,MaxPooling1D
from keras.layers import Bidirectional, GlobalMaxPool1D, Conv2D, SpatialDropout1D, BatchNormalization, GlobalMaxPooling2D,Conv1D
from keras.initializers import glorot_normal
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import initializers, regularizers, constraints, optimizers, layers, Sequential
from keras import backend as K

early_stop = EarlyStopping(monitor="loss", mode="min", patience=3)

model = Sequential()
model.add(Embedding(max_features, embed_size, weights=[embedding_matrix],trainable = True,name = 'Word-Embedding-Layer')) 
model.add(Dropout(0.4,name = 'Dropout-Regularization-1')) # Best = 0.3
model.add(Bidirectional(LSTM(12, return_sequences=True, dropout=0.35, recurrent_dropout=0.35,kernel_initializer=glorot_normal(seed=None)),name = 'BDLSTM')) #Best = 300,0.25,0.25
model.add(GlobalMaxPool1D(name = 'Global-Max-Pool-1d')) 
model.add(Dense(y_binary.shape[1], activation="softmax",name = 'FC-Output-Layer'))
model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['mse', 'acc']) 
history = model.fit(xtrain, ytrain,
    validation_split=0.2, 
    validation_data=(xtest,ytest), 
    batch_size=5000, epochs=1, 
    callbacks=[early_stop],verbose=1)
#history = model.fit(X_train, Y_train, epochs=42, batch_size=50, verbose=1)
print(history.history.keys())
model.save("data/model.h5")


evaluation = model.evaluate(x=xtest, y=ytest, batch_size=1000, verbose=1, sample_weight=None, steps=None)

#from sklearn.metrics import roc_curve
#prediction = model.predict_classes(x=xtrain, batch_size=1000).ravel()
prediction = model.predict(x=xtrain, batch_size=1000).ravel()
#fpr_keras, tpr_keras, thresholds_keras = roc_curve(ytest.argmax(),prediction.argmax())

#from sklearn.metrics import auc
#auc_keras = auc(fpr_keras, tpr_keras)

#import tensorflow as tf
#prediction1=tf.argmax(logits,1)
#print(prediction[0])
#print(prediction)

import numpy as np
np.savetxt('data/predictions.csv', prediction, delimiter=',')

from sklearn.ensemble import RandomForestClassifier
# Supervised transformation based on random forests
rf = RandomForestClassifier(max_depth=3, n_estimators=10)
rf.fit(xtrain, ytrain)

y_pred_rf = rf.predict_proba(xtest)[:, 1]
fpr_rf, tpr_rf, thresholds_rf = roc_curve(ytest, y_pred_rf)
auc_rf = auc(fpr_rf, tpr_rf)


plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
# Zoom in view of the upper left corner.
plt.figure(2)
plt.xlim(0, 0.2)
plt.ylim(0.8, 1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve (zoomed in at top left)')
plt.legend(loc='best')
plt.show()
#print('close terms')
#print(word_vectors.similarity('cash', 'total'))
#print(word_vectors.similarity('bottle', 'deposit'))
#print(word_vectors.similarity('debit', 'card'))
#print(word_vectors.similarity('gift', 'card'))
#print(word_vectors.similarity('sauvignon', 'blanc'))

#print('distant terms')
#print(word_vectors.similarity('gift', 'vodka'))
#print(word_vectors.similarity('gift', 'juice'))
#print(word_vectors.similarity('sauvignon', 'wine'))
#print(word_vectors.similarity('750ml', 'cash'))

#word_vectors =  gensim.models.KeyedVectors.load_word2vec_format('data/word2vec/glove_model2.txt', binary=False)
#vocab = word_vectors.wv.vocab
#print(vocab["relic"])

#print('close terms')
#print(word_vectors.similarity('cash', 'total'))
#print(word_vectors.similarity('bottle', 'deposit'))
#print(word_vectors.similarity('debit', 'card'))
#print(word_vectors.similarity('gift', 'card'))
#print(word_vectors.similarity('sauvignon', 'blanc'))

#print('distant terms')
#print(word_vectors.similarity('gift', 'vodka'))
#print(word_vectors.similarity('gift', 'juice'))
#print(word_vectors.similarity('sauvignon', 'wine'))
#print(word_vectors.similarity('750ml', 'cash'))

#print("Gensim Model Loaded: %s" % gensim_file)
#return(word_vectors)





