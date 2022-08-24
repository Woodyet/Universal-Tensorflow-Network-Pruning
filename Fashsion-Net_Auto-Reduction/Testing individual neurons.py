
# TensorFlow and tf.keras
import tensorflow as tf
from tqdm import tqdm
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
import json
import copy
import pandas as pd
import tensorflow_model_optimization as tfmot
import skimage.measure
import tempfile
from collections import Counter
import time
import random
import string
import os
import statistics

mae = tf.keras.losses.MeanAbsoluteError()
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input

from kerassurgeon.operations import delete_channels

print(tf.__version__)

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def custom_loss(y_actual,y_pred):
    a=mae(y_actual,y_pred)
    b=tf.Variable(0, shape=tf.TensorShape(None))
    return tf.where(y_actual==0,a,b,"Tester")


train_images = train_images / 255.0

test_images = test_images / 255.0

train_images = train_images.reshape((train_images.shape[0], 28, 28,1))
test_images = test_images.reshape((test_images.shape[0], 28, 28,1))

dummy_data = np.zeros((60000,1))
Filters = 64
FC_amm = 128
remove_amount = 0.90
epochs = 3
In = Input(shape=(28,28,1),name="YIN")
x2 = Conv2D(Filters, (3, 3), activation='relu',name="layer_prune2")(In)
x = Flatten()(x2)
x1 = Dense(FC_amm,name="layer_prune")(x)
x = Dense(10,name="labels")(x1)

model = Model([In],[x], name="WOWOWOWOOW")

model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=epochs)

orioki = model.evaluate(test_images,test_labels, verbose=2)
print('\nTest accuracy:', orioki)

postweights = model.get_weights()

modelfrank = Model([In],[x,x1,x2], name="WOWOWOWOOW")

modelfrank.set_weights(postweights)

import math

Layer_Data_stacked = []
Layer_Data2_stacked = []

dataset_size = train_images.shape[0]

splits = 2500

dataset_splits = int(dataset_size/splits)

for i in tqdm(range(dataset_splits)):
    _,Layer_Data,Layer_Data2 = modelfrank.predict(train_images[splits*i:splits+splits*i])
    Layer_Data_stacked.append(Layer_Data)
    Layer_Data2_stacked.append(Layer_Data2)
print("stacking")
Layer_Data = np.concatenate(Layer_Data_stacked)
Layer_Data_stacked = 0
Layer_Data2 = np.concatenate(Layer_Data2_stacked)
Layer_Data2_stacked = 0
print("stacked")
import random
print("Making data")


FCWeights = []
for i in range(FC_amm):
    FCWeights.append(0)
ConvWeights = []
for i in range(Filters):
    ConvWeights.append(0)

num_list1 = random.sample(range(0, Filters), int(Filters*remove_amount))  #64

num_list1.sort()
print("1")
for i in tqdm(num_list1):
    Layer_Data2[:,:,:,i] = 0
    #ConvWeights[i] = 4

num_list2 = random.sample(range(0, FC_amm), int(FC_amm*remove_amount))

num_list2.sort()
print("2")
for i in tqdm(num_list2):
    Layer_Data[:,i] = 0
    #FCWeights[i] = 2

losses = {"labels": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),"layer_prune": 'mean_absolute_error',"layer_prune2": 'mean_absolute_error'}
#lossWeights = {"labels": 1/100, "layer_prune": 89/100,"layer_prune2": 10/100}

lossWeights = {"labels": 1, "layer_prune": 2,"layer_prune2": Filters}#int(Filters*remove_amount)

classweights = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
FCWeights
ConvWeights

#lossWeights = [classweights,FCWeights,ConvWeights]

modelfrank.compile(optimizer='adam',
                  loss=losses,
                  loss_weights=lossWeights,
                  metrics=['accuracy'])



oki = model.evaluate(test_images,test_labels, verbose=2)
print('\nTest accuracy:', oki)


modelfrank.fit(train_images, [train_labels,Layer_Data,Layer_Data2], epochs=epochs)

#_,Layer_Data = modelfrank.predict(test_images)
#Layer_Data[:,0] = 0
#oki = modelfrank.evaluate(test_images,[test_labels,Layer_Data], verbose=2)
#print('\nTest accuracy:', oki)

model.set_weights(modelfrank.get_weights())
model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

oki = model.evaluate(test_images,test_labels, verbose=2)

sooo = modelfrank.predict(test_images[:5])

t = sooo[2][:,:,:,0]
v = sooo[2][:,:,:,1]
r = sooo[2][:,:,:,2]


model = delete_channels(model, model.get_layer(name="layer_prune"), num_list2)
model = delete_channels(model, model.get_layer(name="layer_prune2"), num_list1)

model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

oki2 = model.evaluate(test_images,test_labels, verbose=2)

model.fit(train_images, train_labels, epochs=epochs)

oki3 = model.evaluate(test_images,test_labels, verbose=2)

print('\nOri test accuracy:', orioki)
print('\nUnPruned Test accuracy:', oki)
print('\nPruned Test accuracy:', oki2)
print('\nPrunedRT Test accuracy:', oki3)
temp = 0