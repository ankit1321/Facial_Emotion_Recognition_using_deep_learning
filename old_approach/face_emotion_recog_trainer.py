# USAGE
# python Face_emotion_recog_trainer.py --dataset dataset


# import the necessary packages
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.models import load_model
# from keras.layers import LeakyReLU
from sklearn.preprocessing import LabelBinarizer,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str,
	default="Face_recognizer",
	help="path to output face mask detector model")
args = vars(ap.parse_args())

# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 1e-3
EPOCHS = 10
BS = 128

# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []
# loop over the image paths
# count=1
for imagePath in imagePaths:
	# extract the class label from the filename
	label = imagePath.split(os.path.sep)[-2]

	# load the input image (224x224) and preprocess it
	image = load_img(imagePath, target_size=(48, 48))
	image = img_to_array(image)
	image = preprocess_input(image)
	# update the data and labels lists, respectively
	# count+=1/
	# if count%10==0:
	data.append(image)
	labels.append(label)

# convert the data and labels to NumPy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)
print("labels",labels.shape)
# perform one-hot encoding on the labels
lb = LabelEncoder()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)
print(labels,type(labels))

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
print(data.shape,labels.shape)
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

# construct the training image generator for data augmentation
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# load the MobileNetV2 network, ensuring the head FC layer sets are
# left off
# baseModel = MobileNetV2(weights="imagenet", include_top=False,
# 	input_tensor=Input(shape=(224, 224, 3)))
# baseModel = Xception(weights="imagenet",include_top=False,
# 					 input_tensor=Input(shape=(48, 48, 3)))
# print(baseModel)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer,Conv2D,MaxPool2D
# from tensorflow.keras.callbacks import ModelCheckpoint
# monitor = ModelCheckpoint("checkpoint",verbose=2,save_best_only=True)
baseModel = Sequential()
# baseModel.add(Dense(12, activation='relu', input_tensor = Input(shape=(48,48,3))))
baseModel.add(InputLayer(input_shape = (48,48,3)))
headModel = baseModel.output
headModel = Conv2D(96,3,activation="relu",padding = "same")(headModel)
headModel = Conv2D(96,3,activation="relu")(headModel)
headModel = MaxPool2D(pool_size = (2,2),padding = "valid")(headModel)
headModel = Conv2D(48,3,activation="relu",padding = "same")(headModel)
headModel = Conv2D(48,3,activation="relu")(headModel)
headModel = MaxPool2D(pool_size = (2,2),padding = "valid")(headModel)
headModel = Conv2D(48,3,activation="relu",padding = "same")(headModel)
headModel = Conv2D(48,3,activation="relu")(headModel)
headModel = MaxPool2D(pool_size = (2,2),padding = "valid")(headModel)
# baseModel=load_model('mask_detector.model')

# construct the head of the model that will be placed on top of the
# the base model
# headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(1, 1))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(112, activation="relu")(headModel)
headModel = Dense(56, activation="tanh")(headModel)
headModel = Dropout(0.15)(headModel)
headModel = Dense(28, activation="relu")(headModel)
headModel = Dense(28, activation="tanh")(headModel)
headModel = Dropout(0.2)(headModel)
headModel = Dense(14, activation="relu")(headModel)
headModel = Dense(14, activation="tanh")(headModel)
headModel = Dense(7, activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = True

# compile our model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the head of the network
print("[INFO] training head...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	# callbacks = [monitor],
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

# from tensorflow.keras.models import load_model
# model = load_model("checkpoint/")

# make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)
print(predIdxs)
# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)
print(predIdxs)

# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# serialize the model to disk
print("[INFO] saving mask detector model...")
model.save(args["model"], save_format="h5")

history_dict = H.history
print(history_dict.keys())

# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
