#Project reference: https://data-flair.training/blogs/python-deep-learning-project-handwritten-digit-recognition/

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

#splits MNIST data into training and test sets, already pre-formatted data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)

#reshape data sets to dimensions of (60000, 28, 28, 1)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
#28 channels, 28 rows, 1 column
input_shape = (28, 28, 1)

batch_size = 128 
num_classes = 10 #number of classes for data classification
epochs = 10

#convert vectors to binary class matrix, more info here: https://stackoverflow.com/questions/61307947/whats-binary-class-matrix-in-context-of-deep-learning
#and here: https://www.rdocumentation.org/packages/kerasR/versions/0.6.1/topics/to_categorical
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

#Creating the Convolution Neural Network (CNN), includes multiple layers and
#works well with data represented by grid structures

model = Sequential() #https://keras.io/api/models/sequential/

#creates a convolution kernel
#32 filters, 3x3 kernel size for filtering, ReLU linear activation function
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape)) #https://keras.io/api/layers/convolution_layers/convolution2d/
model.add(Conv2D(64, (3, 3), activation='relu'))
#taking the maximum data over the pool size, outputs same as input: https://keras.io/api/layers/pooling_layers/max_pooling2d/
model.add(MaxPooling2D(pool_size=(2, 2)))
#drops 25% of input units by setting to 0: https://keras.io/api/layers/regularization_layers/dropout/
model.add(Dropout(0.25))
#flattens the input (creates the product of layers): https://keras.io/api/layers/reshaping_layers/flatten/
model.add(Flatten())
#increases (denses up) the dimensionality of the output: https://keras.io/api/layers/core_layers/dense/
model.add(Dense(256, activation='relu'))
#dropping out 50% of the input units
model.add(Dropout(0.5))
#increasing dimensionality to the number of classes, more on softmax function (turns digits into probabilities adding up to 1)
model.add(Dense(num_classes, activation='softmax'))

#computes "cross-entropy loss" and uses an optimizer, Adadelta, which utilizes the stochastic gradient descent method
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])

#training the model, model.fit()
hist = model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))
print("The model has successfully trained")

model.save('mnist.h5')
print("Saving the model as mnist.h5")

#TESTING our model (with the test set)
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])