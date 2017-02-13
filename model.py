from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout,  Lambda
from keras.layers.convolutional import Convolution2D, Convolution1D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2
import csv
import random
from random import randint
from PIL import Image
import numpy as np
import sys
import matplotlib.pyplot as plt
from collections import Counter

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf

filepath = "udadata"
datalist = []

correction=0.26
##read csv
#columns 'center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed
def read_csv():
	data = []
	with open(filepath+'/driving_log.csv', newline='') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		print("Reading Images\n")
		for row in reader:
			if row[0] == 'center': 
				continue
			data.append(row)
		return data			


#get train data only center camera images
def get_train_data(data):
	x = []
	y = []
	for i in data:
		x.append(i[0])#image path
		y.append(float(i[3]))#steering angle
		
	return x,y
	
#create some random brightness on the image	
def randomize_brightness(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    rand = random.uniform(0.3,1.0)
    image[:,:,2] = rand*image[:,:,2]
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    return image	

def cvcrop(img):
	shape=img.shape
	return img[60:shape[0]-20, 0:shape[1]]
	
def cvflip(img, angle):
	img = cv2.flip(img,1)
	angle = angle * (-1)
	return img, angle

def data_generator(size,epochs):
	x = []
	y = []
	while 1:
		#reshuffle data every data generation
		x_s,y_s = shuffle(X_train, y_train)
		count=0
		while 1:	
			random_i = randint(0,len(X_train)-1)
			
			image = cv2.imread(filepath+'/'+ x_s[random_i].strip())		
			#crop image
			image = cvcrop(image)
			#resize 64x64
			image = cv2.resize(image, (64,64))
			
			angle=y_s[random_i]				
			
			count += 1	
			
			#flip images randomly
			flip=randint(0,1)
			if flip == 1:
				image,angle=cvflip(image,angle)			
			
			image = randomize_brightness(image)			
			x.append(image)
			try:#add some random angles modifications
				y.append(angle * (1+ np.random.uniform(-0.11,0.11)))
			except TypeError:
				print(angle)		
			
			if count >= size:
				break
			
		x_batch = np.asarray(x,dtype=np.float32)
		y_batch = np.asarray(y,dtype=np.float32)
		x = []
		y = []
		yield(x_batch,y_batch)
		
def get_balanced_data(data):
	x = []
	y = []
	not_balanced = 1
	maxcount = 1
	while not_balanced:
		for i in data:
			angle = float(i[3])
			
			if y.count(angle) >= maxcount:
				continue
			else:
				x.append(i[0])#image path
				y.append(angle)#steering angle
				
			not_balanced=0
			if len(y)>0:
				for j in y:
					if y.count(j) < maxcount:
						not_balanced = 1
						
	return x,y

def get_wandering_simulation_data5(x,y):
	x_ = []
	y_ = []
	for i in range(len(y)):
		angle=y[i]
		if angle > 0.12:
			for z in range(1):
				x_.append(x[i])#left image path
				y_.append(angle + correction)#correct steering angle
				
				continue
		
		if angle < -0.12:
			for z in range(1):
				x_.append(x[i])#right image path
				y_.append(angle - correction)#correct steering angle				
				continue
				
	return x_,y_
	
def modelA():
	shape = (64,64,3)

	#nvidia model based network

	model = Sequential()
	model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape = shape))
	model.add(Convolution2D(24, 5, 5, subsample=(2,2), border_mode="valid", init='he_normal'))
	model.add(Activation('relu'))
	model.add(Convolution2D(36, 5, 5, subsample=(2,2), border_mode="valid",init='he_normal'))
	model.add(Activation('relu'))
	model.add(Convolution2D(48, 5, 5, subsample=(2,2), border_mode="valid",init='he_normal'))
	model.add(Activation('relu'))
	model.add(Convolution2D(64, 3, 3, subsample=(1,1), border_mode="valid",init='he_normal'))
	model.add(Activation('relu'))
	model.add(Convolution2D(64, 3, 3, subsample=(1,1), border_mode="valid",init='he_normal'))
	model.add(Activation('relu'))
	#model.add(Dropout(0.5))


	model.add(Flatten())
	model.add(Dense(256))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(100))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(64))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(16))
	model.add(Activation('relu'))
	model.add(Dense(1))
	
	return model
	
def modelB():
	model = Sequential()
	model.add(Lambda(lambda x: x/127.5 - 1.0,input_shape=(64,64,3)))
	model.add(Convolution2D(32, 8,8 ,border_mode='same', subsample=(4,4)))
	model.add(Activation('relu'))
	model.add(Convolution2D(64, 8,8 ,border_mode='same',subsample=(4,4)))
	model.add(Activation('relu',name='relu2'))
	model.add(Convolution2D(128, 4,4,border_mode='same',subsample=(2,2)))
	model.add(Activation('relu'))
	model.add(Convolution2D(128, 2,2,border_mode='same',subsample=(1,1)))
	model.add(Activation('relu'))
	model.add(Flatten())
	model.add(Dropout(0.5))
	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(128))
	model.add(Dense(1))
	model.summary()
	
	return model
	
def modelC():
	shape = (64,64,3)
	model = Sequential()
	model.add(Convolution2D(32, 3, 3, input_shape = shape, border_mode='same', activation='relu'))
	model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
	model.add(Dropout(0.5))
	model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
	model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
	model.add(Dropout(0.5))
	model.add(Flatten())
	model.add(Dense(1024, activation='relu'))
	model.add(Dense(512, activation='relu'))
	model.add(Dense(128, activation='relu'))
	model.add(Dense(1, name='output', activation='tanh'))
	
	return model


datalist = read_csv()

#print(len(datalist))

#balance de data set
X_train, y_train = get_balanced_data(datalist)

#upsample the balance data
X_train = X_train * 100
y_train = y_train * 100

#create recovery data
X_train_2, y_train_2 = get_wandering_simulation_data5(X_train,y_train)

#Add recovery data to train data
X_train = X_train + X_train_2  #+ X_train_3
y_train = y_train + y_train_2  #+ y_train_3

#shuffle data
X_train, y_train = shuffle(X_train, y_train)

#print(len(X_train))
#print(len(y_train))

#create model to use
model = modelA()

#optimizer
adam = Adam(lr = 0.00001)# beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

#compile model
model.compile(loss='mse', optimizer=adam)

batch=128
epochs=15


#train model
history = model.fit_generator(data_generator(batch,epochs), samples_per_epoch=len(X_train), nb_epoch=epochs, max_q_size=1, nb_worker=1)

# serialize model to JSON
print("Writing Model json")
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
    
print("Writing Model h5")
model.save_weights('model.h5')



