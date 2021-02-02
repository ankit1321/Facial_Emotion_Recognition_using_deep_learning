# load json and create model
# from __future__ import division
from keras.models import load_model
# from keras.layers import Dense
# from keras.models import model_from_json
import argparse
import numpy
import os
import numpy as np
import cv2


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
    help="path to where the face cascade resides")
ap.add_argument("-m", "--model", required=True,
    help="path to pre-trained emotion detector ")
ap.add_argument("-i", "--image", required=True,
    help="path to the image file")
args = vars(ap.parse_args())


loaded_model = load_model(args["model"])
#setting image resizing parameters
WIDTH = 48
HEIGHT = 48
x=None
y=None
labels = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

#loading image
image = cv2.imread(args["image"])
print("Image Loaded")
gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
face_detector = cv2.CascadeClassifier(args["cascade"])
faces = face_detector.detectMultiScale(gray, 1.2  , 5)
print(len(faces))
canvas = np.zeros((220, 300, 3), dtype="uint8")

#detecting faces
if len(faces):
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        cv2.imwrite('cropped.jpg',roi_gray)
        fin_roi_gray = cv2.imread('cropped.jpg')
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(fin_roi_gray, (48, 48)), -1), 0)
        cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2,
                      dtype=cv2.CV_32F)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
        #predicting the emotion
        yhat= loaded_model.predict(cropped_img)
        cv2.putText(image, labels[int(np.argmax(yhat))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 1, cv2.LINE_AA)
        print("Emotion: "+labels[int(np.argmax(yhat))])
        os.remove('cropped.jpg')
        for (i, (label, prob)) in enumerate(zip(labels, yhat[0])):
            # construct the label text
            text = "{}: {:.2f}%".format(label, prob * 100)
            w = int(prob * 300)
            cv2.rectangle(canvas, (5, (i * 35) + 5),(w, (i * 35) + 35), (0, 0, 255), -1)
            cv2.putText(canvas, text, (10, (i * 35) + 23),cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 2)
            cv2.imshow("Probabilities", canvas)

    cv2.imshow('Emotion', image)
    cv2.waitKey()
else:
    print('can not detect face in image')