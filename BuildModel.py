from google.colab import drive 
drive.mount('/content/gdrive')

!unzip gdrive/My\ Drive/EPQDataset/CMFD.zip -d /content/Correctly-Masked

!unzip gdrive/My\ Drive/EPQDataset/IMFD.zip -d /content/Incorrectly-Masked

import numpy as np 
from PIL import Image
import os as os 
import random as random
import tensorflow as tf 
from keras import models, datasets, layers

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


def formatImage(path):
  img = Image.open(path).resize((256, 256))
  return np.array(img)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

for i in range(0, 10):
  
  data, labels = createBatch(i)
 
  
  x = model.fit(data, labels, epochs=10)

data, labels = createBatch(34)

model.evaluate(data, labels, verbose = 2)

model.save("model2.h5")