
import numpy as np 
from PIL import Image
import os as os 
import random as random
import tensorflow as tf 
from keras import models, datasets, layers
from keras.applications import MobileNetV2

from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, AveragePooling2D, Dropout
from keras.models import Model

d = ["D:\Dataset\IMFD1", "D:\Dataset\CMFD"]

def createBatch(fileNum):
  dataSet = []
  labels = []
  directories = [os.path.join(d[0], os.listdir(d[0])[fileNum]), os.path.join(d[1], os.listdir(d[1])[fileNum])]
  print(directories)
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


def formatImage(path):
  img = Image.open(path).resize((224, 224))
  return np.array(img)

base = MobileNetV2(weights="imagenet", include_top=False,
	input_shape=(224, 224, 3))

top = base.output
top = AveragePooling2D(pool_size=(7, 7))(top)
top = Flatten(name="flatten")(top)
top = Dense(128, activation="relu")(top)
top = Dropout(0.5)(top)
top = Dense(2, activation="softmax")(top)

model = Model(inputs=base.input, outputs=top)

for layer in base.layers:
  layer.trainable = False

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer="adam",
	metrics=["accuracy"])

model.summary()

for i in range(0, 12):
  
  data, labels = createBatch(i)



  x = model.fit(data, labels, epochs=10)

data, labels = createBatch(34)

model.evaluate(data, labels, verbose = 2)

model.save("model4.h5")

