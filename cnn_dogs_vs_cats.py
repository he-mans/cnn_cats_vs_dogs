import cv2 as opencv
from tensorflow import keras
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Activation,Flatten
import os
import numpy as np
import time
import random

def generate_data(dataset):
	'''giving location of cats and dogs dataset'''
	cats = 'dataset/training_set/cats'
	dogs = 'dataset/training_set/dogs'

	'''array of pixel values for cats and dogs'''
	cats_array, dogs_array = [],[]
	image_size = 100
	cat_classification_index = 0
	dog_classification_index = 1

	cat_images = os.listdir(cats)
	dog_images = os.listdir(dogs)
	os.chdir(cats)
	print("started processing data")
	current = time.time()

	for cat,dog in zip(cat_images,dog_images):
		'''opening image in gray scale and appending their pixel values to 
			respective arrays with their classification index'''
		os.chdir("../cats")
		image = opencv.imread(cat,opencv.IMREAD_GRAYSCALE)
		image = opencv.resize(image,(image_size,image_size))
		cats_array.append([image,cat_classification_index])
		
		os.chdir("../dogs")
		image = opencv.imread(dog,opencv.IMREAD_GRAYSCALE)
		image = opencv.resize(image,(image_size,image_size))
		dogs_array.append([image,dog_classification_index])

	'''combining separate datasets in to one dataset'''
	dataset+= cats_array+dogs_array
	random.shuffle(dataset)

	print("precessing ended")
	print(f"total time taken {time.time()-current} seconds")
	os.chdir("../../..")

def get_validation_data(dataset_validation):
	cats = 'dataset/test_set/cats'
	dogs = 'dataset/test_set/dogs'

	cats_images = os.listdir(cats)
	dogs_images = os.listdir(dogs)
	os.chdir(cats)
	
	cats_array, dogs_array = [],[]
	image_size = 100
	cat_classification_index=0
	dog_classification_index=1
	
	for cat,dog in zip(cats_images,dogs_images):
		os.chdir("../cats")
		image = opencv.imread(cat,opencv.IMREAD_GRAYSCALE)
		image = opencv.resize(image,(image_size,image_size))
		cats_array.append([image,cat_classification_index])

		os.chdir("../dogs")
		image = opencv.imread(dog,opencv.IMREAD_GRAYSCALE)
		image = opencv.resize(image,(image_size,image_size))
		dogs_array.append([image,dog_classification_index])

	dataset_validation+=cats_array+dogs_array
	random.shuffle(dataset)
	os.chdir("../../..")

def train_model(dataset,dataset_validation):
	lable, image,image_validation,lable_validation = [],[],[],[]
	image_size = 100
	
	for data in dataset:
		image.append(data[0])
		lable.append(data[1])
	
	for data in dataset_validation:
		image_validation.append(data[0])
		lable_validation.append(data[1])
	
	#converting some testing data to training data for better result
	image+=image_validation[1000:]
	lable+=lable_validation[1000:]
	image_validation=image_validation[:1000]
	lable_validation=lable_validation[:1000]
	
	#converting data list into numpy arrays
	image = np.array(image).reshape(len(image),image_size,image_size,1)/255.0
	lable = np.array(lable).reshape(len(lable),1)
	
	image_validation = np.array(image_validation).reshape(len(image_validation),image_size,image_size,1)/255.0
	lable_validation = np.array(lable_validation).reshape(len(lable_validation),1)

	model = keras.models.Sequential()

	model.add(Conv2D(64,(3,3),strides=1,use_bias=True,activation='relu',padding='same',input_shape=(image_size,image_size,1)))
	model.add(MaxPooling2D(pool_size=(2,2),strides=2))

	model.add(Conv2D(64,(3,3),strides=1,activation='relu',padding='same',use_bias=True))
	model.add(MaxPooling2D(pool_size=(2,2),strides=2))

	model.add(Flatten())
	model.add(Dense(64,activation='relu'))
	model.add(Dense(32,activation='relu'))
	model.add(Dense(1,activation='sigmoid'))

	model.compile(optimizer='adam',
					loss = 'binary_crossentropy',
					metrics = ['accuracy'])
	model.fit(image,lable,batch_size=30,epochs=4,validation_data=(image_validation,lable_validation))

	model.save('dogs_vs_cats.model')



if __name__ == "__main__":
	dataset = []
	dataset_validation = []
	#passing dataset by reference to generate dataset
	generate_data(dataset)
	get_validation_data(dataset_validation)
	#training model
	train_model(dataset,dataset_validation)

