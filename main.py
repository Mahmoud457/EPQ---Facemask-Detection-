import numpy
import tensorflow as tf
from keras import models, datasets, layers
import numpy as np
import os as os
from PIL import Image
import cv2 as cv2
import pygame




categories = ["Not Masked", "Masked"]

def grabImage():

    if not pic.isOpened():
        raise IOError("Cannot open webcam")
    ret, image = pic.read()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    return image
def formatImage(image):
    img = Image.fromarray(image).resize((winWidth, winHeight))
    return np.array(img)

def displayImage(image):
    raw = image.tobytes() #Converts to datatype readable by Pygame
    surface = pygame.image.frombuffer(raw, image.shape[:2], "RGB")
    window.blit(surface, (0,0))

def Predict(image):
    prediction = model.predict(image)
    print(prediction)
    return categories[np.argmax(prediction)] #takes a number of predictions and only shows one with highest probability 
def Check(res):
    if np.all(res == res[0]):
        return True
    return False

def Write(msg):
    font = pygame.font.Font('freesansbold.ttf', 32)
    text = font.render(msg, True, (255, 255, 255), (0,0,0))
    textRect = text.get_rect()
    textRect.center = (winWidth//2, winHeight//2)
    window.blit(text, textRect)





pygame.init()


max = 5
winWidth, winHeight = 224, 224
count = 0

model = models.load_model("model9.h5")
model.summary()
pic = cv2.VideoCapture(0)
results = []
window = pygame.display.set_mode((winWidth, winHeight))
pygame.display.set_caption('Facemask Detection')
pygame.display.flip()



run = True
while run:

    img = formatImage(grabImage())
    displayImage(img)

    results.append(Predict(np.array([img])))

    if count == (max-1):

        if Check(np.array(results)):
            Write(results[0])
            print(results[0])
        count = 0
        results = []
    else:
        count+=1


    pygame.display.flip()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

pic.release()
cv2.destroyAllWindows()
