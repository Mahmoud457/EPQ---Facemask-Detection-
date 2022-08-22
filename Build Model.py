# -*- coding: utf-8 -*-
"""EPQ pre trained model

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1hsE5eiiNNs2NxQdRWodnD6vIQC9w3OcI
"""

from google.colab import drive 
drive.mount('/content/gdrive')

!unzip gdrive/My\ Drive/EPQ/EPQDataset/CMFD.zip -d /content/Correctly-Masked

!unzip gdrive/My\ Drive/EPQ/EPQDataset/IMFD.zip -d /content/Incorrectly-Masked

import numpy as np
from PIL import Image
import os as os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from keras.applications.mobilenet_v2 import MobileNetV2
from keras import models, datasets, layers
import random

d = ["/content/Incorrectly-Masked", "/content/Correctly-Masked"]




def createBatch(fileNum):
  dataSet = []
  labels = []
  directories = [os.path.join(d[0], os.listdir(d[0])[fileNum]), os.path.join(d[1], os.listdir(d[1])[fileNum])]
  lab = 0
  for dir in directories:
    for image in os.listdir(dir):
        dataSet.append(formatImage(os.path.join(dir, image)))
        labels.append(lab)
    lab = 1
  i = random.random()
  random.seed(i)
  random.shuffle(dataSet)
  random.seed(i)
  random.shuffle(labels)
  return np.array(dataSet), np.array(labels)
  return np.array(dataSet), np.array(labels)


def formatImage(path):
  img = Image.open(path).resize((224, 224))
  return np.array(img)

base = MobileNetV2(weights="imagenet", include_top=False,
                   input_shape=(224, 224, 3))


top = layers.Rescaling(1./255)
top = layers.RandomZoom([-0.5, 0.5])
top = layers.RandomRotation(0.2)

top = base.output
top = layers.AveragePooling2D(pool_size=(7, 7))(top)
top = layers.Flatten(name="flatten")(top)
top = layers.Dense(128, activation="relu")(top)
top = layers.Dropout(0.5)(top)
top = layers.Dense(2, activation="softmax")(top)

model = models.Model(inputs=base.input, outputs=top)

for layer in base.layers:
  layer.trainable = False

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam",
              metrics=["accuracy"])

model.summary()

for i in range(0, 25):
  
  data, labels = createBatch(i)
 
  
  x = model.fit(data, labels, epochs=6)

data, labels = createBatch(34)

model.evaluate(data, labels, verbose = 2)

model.save("model9.h5")

from google.colab import files

files.download("/content/model9.h5")