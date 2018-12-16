from tensorflow import keras
import os
import cv2 as opencv
import random
import numpy as np

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
	random.shuffle(dataset_validation)
	os.chdir("../../..")

def predict(dataset_validation):
	image_validation,lable_validation=[],[]

	for data in dataset_validation:
		image_validation.append(data[0])
		lable_validation.append(data[1])

	count = 0
	#last 1000 images were used as surplus training data
	image_validation = image_validation[:1000]
	lable_validation = lable_validation[:1000]
	image_size=100
	image_validation = np.array(image_validation).reshape(len(image_validation),image_size,image_size,1)/255.0

	model = keras.models.load_model('dogs_vs_cats.model')
	loss,accuracy = model.evaluate(image_validation,lable_validation,batch_size=30)
	print(f"loss : {loss}\naccuracy : {accuracy}")

if __name__ == '__main__':
	dataset_validation=[]
	get_validation_data(dataset_validation)
	predict(dataset_validation)
