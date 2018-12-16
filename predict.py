from tensorflow import keras
import numpy as np
import cv2 as opencv

print('enter image name (only jpg image allowed)')
name = input()

image_size = 100
image = opencv.imread(name,opencv.IMREAD_GRAYSCALE)
image = opencv.resize(image,(image_size,image_size))
image = np.array(image).reshape(1,image_size,image_size,1)

model = keras.models.load_model('dogs_vs_cats.model')
prediction = model.predict(image)
print(prediction)

if prediction > .5:
	print('dog')
else:
	print("cat")
