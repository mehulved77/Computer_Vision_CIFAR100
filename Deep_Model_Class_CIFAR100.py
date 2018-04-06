from __future__ import absolute_import, print_function
import os
import sys
import tensorflow as tf
from functools import wraps
import numpy as np
from pprint import pprint
from time import time
import keras as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

class DeepModel:
    
    def timed(f):
        @wraps(f)
        def wrapper(*args, **kwds):
            start = time()
            result = f(*args, **kwds)
            elapsed = time() - start
            #print("%s took %d time to finish" % (f._name_, elapsed))
            return(result)
        return wrapper
    
    def __init__(self, x_train, y_train, x_test, y_test, num_classes, input_shape):
        print('[INFO] Python Version: {}'.format(sys.version_info[0]))
        print('[INFO] TensorFlow Version: {}'.format(tf.__version__))
        print('[INFO] Keras Version: {}'.format(K.__version__))
        print('[INFO] GPU Enabled?: {}'.format(tf.test.gpu_device_name() is not ''))
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.num_classes = num_classes
        self.input_shape = input_shape
         
       
    def __repr__(self):
         return "Deep Model Object"
    
    
    
    def cnn(self):
    # create model
        model = Sequential()
        model.add(Conv2D(30, (5, 5), input_shape=self.input_shape, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(15, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2))) 
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(self.num_classes, activation='softmax'))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model = model
        return self

    @timed
    def train(self, batch_size, epochs):
        print("[INFO] Starting training with {} epochs...".format(epochs))
        self.model.fit(self.x_train, self.y_train, 
          validation_data=(self.x_test,self.y_test), 
          epochs=epochs, batch_size = batch_size, verbose=2)
        return self
    
    @timed
    def predict(self):
        print("[INFO] Starting prediction on {}...")
        self.scores = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print("CNN Accuracy: {:.2f}%".format(self.scores[1]*100))
        print("CNN Error: {:.2f}%".format(100-self.scores[1]*100))
        return self
    
if __name__ == "__main__":
    from data_cifar100 import x_train, y_train, x_test, y_test, batch_size, num_classes, input_shape
    newmodel = DeepModel(x_train, y_train, x_test, y_test, num_classes, input_shape)
    newmodel.cnn().train(128,1).predict()
    