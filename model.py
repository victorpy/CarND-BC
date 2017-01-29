from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, Convolution1D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from sklearn.utils import shuffle
import csv
import random
from random import randint
from PIL import Image
import numpy as np
import sys
import matplotlib.pyplot as plt

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf


#features = []
#labels = []
#resize=(320, 160)
resize=(160,80)

def normalize_grayscale(image_data):
    a = -0.5
    b = 0.5
    grayscale_min = 0
    grayscale_max = 255
    return a + ( ( (image_data - grayscale_min)*(b - a) )/( grayscale_max - grayscale_min ) )
    
def shuffle_train_file():
	 #shuffle file lines
	print("shuffling train file")
	lines = open('data3/driving_log.csv').readlines()
	random.shuffle(lines)
	open('data3/driving_log.csv', 'w').writelines(lines)

def shuffle_val_file():
	print("shuffling val file")
	lines = open('traindata5/driving_log.csv').readlines()
	random.shuffle(lines)
	open('traindata5/driving_log.csv', 'w').writelines(lines)


			
def generate_images_from_file(filepath, maxim):
	
	while 1:
		features = []
		labels = []
		count = 0;
		
		with open(filepath+'/driving_log.csv', newline='') as csvfile:
			reader = csv.reader(csvfile, delimiter=',')
			print("Reading Images\n")			
			for row in reader:
				if row[0] == 'center': 
					continue
				#print(row[0]+' '+row[3])
				for j in range(0,3):
				#print(row[0]+' '+row[3])
				#image = Image.open(filepath+'/'+row[randint(0,2)].strip())
					image = Image.open(filepath+'/'+row[j].strip())
					image = image.resize(resize, Image.BICUBIC)
					rotate=randint(0,10)
					#print(rotate)
					if rotate >= 6:
						#print("rotated")
						image=image.rotate(180, resample=Image.BICUBIC)
				#gray=randint(0,10)
				#print(rotate)
				#if gray >= 6:
					#print("rotated")
					#image=image.convert("L")
				data = np.asarray( image, dtype="uint8" )
				features.append(data)
				labels.append(round(float(row[3]), 4))
				count += 1
					#if count >= maxim:
					#	break
				#yield (x , y)
				if count >= maxim:
					#print("in count "+str(count))
					x = np.asarray(features)
					x = normalize_grayscale(x)
					y = np.asarray(labels, dtype=np.float32)
					#print(y)
					#print(row[0]+' '+str(x.shape))
					x, y = shuffle(x, y)
					features = []
					labels = []
					count = 0					
					yield (x , y)
					#break
		x = np.array(features)
		y = np.array(labels)
		x, y = shuffle(x, y)
		yield (x , y)	
					
					
def generate_val_images_from_file(filepath, maxim):
	
	while 1:
		features = []
		labels = []
		count = 0;
		#shuffle_train_file()
		with open(filepath+'/driving_log.csv', newline='') as csvfile:
			reader = csv.reader(csvfile, delimiter=',')
			print("Reading Images\n")			
			for row in reader:
				if row[0] == 'center': 
					continue
				
				#for j in range(0,3):
					#print(row[j]+' '+row[3])
				#print(row[0]+' '+row[3])
				image = Image.open(filepath+'/'+row[0].strip())
				image = image.resize(resize, Image.ANTIALIAS)
				data = np.asarray( image, dtype="uint8" )
				features.append(data)
				labels.append(round(float(row[3]), 4))
				count += 1
					#if count >= maxim:
					#	break	
				#yield (x , y)
				if count >= maxim:								
					x = np.asarray(features)
					x = normalize_grayscale(x)
					y = np.asarray(labels, dtype=np.float32)
					#print(y)
					#print(row[0]+' '+str(x.shape))
					x, y = shuffle(x, y)
					features = []
					labels = []
					count = 0
					yield (x , y)
					#break

		x = np.array(features)
		y = np.array(labels)
		x, y = shuffle(x, y)
		yield (x , y)	

#shape = (160,320,3)
shape = (80,160,3)

model = Sequential()
model.add(Convolution2D(32, 5, 5, input_shape=shape))
model.add(Activation('relu'))
model.add(Convolution2D(32, 5, 5, input_shape=shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64,3,3))
model.add(Activation('relu'))
model.add(Convolution2D(64,3,3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#model.add(Convolution2D(32,3,3))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

#model.add(Convolution2D(8, 3, 3))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.25))
#model.add(Dense(64))
#model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
#model.add(Dropout(0.5))
model.add(Dense(1))
#model.add(Activation('sigmoid'))


#Compile and train the model
#X_train, y_train = shuffle(X_train, y_train)

#model.compile('adam', 'sparse_categorical_crossentropy', ['accuracy'])
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
sgd = SGD(lr=0.0001, momentum=0.9)
#model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
model.compile(loss='mse', optimizer=sgd, metrics=['accuracy'])
#model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
#model.compile(loss='mse', optimizer='adamax', metrics=['accuracy'])
#model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])

filepath="model.h5"

shuffle_train_file()
shuffle_val_file()

checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]



trainlen = 32
vallen=6
epochs=1000

#history = model.fit_generator(generate_images_from_file('data3',128), validation_data=generate_images_from_file('val_data',32), samples_per_epoch=128, nb_val_samples=32, nb_epoch=30, max_q_size=3, callbacks=callbacks_list)
history = model.fit_generator(generate_images_from_file('data3',trainlen), validation_data=generate_val_images_from_file('traindata5',vallen), samples_per_epoch=trainlen, nb_val_samples=vallen, nb_epoch=epochs, max_q_size=1, nb_worker=1, callbacks=callbacks_list)

# serialize model to JSON
print("Writing Model json")
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
    
print("Writing Model h5")
model.save_weights('model2.h5')


print(history.history['val_acc'][-1])
#print(history.history)
