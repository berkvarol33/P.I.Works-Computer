import numpy as np
import tensorflow as tf
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.neural_network import MLPClassifier
from pandas import read_excel as rc
from pandas import DataFrame as df
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
from seaborn import kdeplot
from sklearn.metrics import confusion_matrix
from seaborn import set, heatmap
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# import warnings
import warnings
# filter warnings
warnings.filterwarnings('ignore')

#Loading the data
dataset_init=rc("dataset3.01.xlsx")
dataset=dataset_init.values
X=dataset[:,100:500]  #Attributes
Y=dataset[:,0:99] #True outputs
X = X.astype(float)
Y = Y.astype(float)

 #Spliting data
X_train=dataset[0:1250,100:300] #100-200 =>escat
X_test=dataset[1250:5001,100:300]
Y_train=dataset[0:1250,0:100]
Y_test=dataset[1250:5001,0:100]

X_train = X_train.reshape(-1,10,20,1)
X_test = X_test.reshape(-1,10,20,1)
# visualize number of digits classes
print("x_train shape",X_train.shape)
print("x_test shape",X_test.shape)
print("y_train shape",Y_train.shape)
print("y_test shape",Y_test.shape)
# 
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

model = Sequential()
#
model.add(Conv2D(filters = 8, kernel_size = (3,1),padding = 'Same', 
                 activation ='softmax', input_shape = (10,20,1)))
model.add(MaxPool2D(pool_size=(3,10)))
model.add(Dropout(0.25))
#
#model.add(Conv2D(filters = 16, kernel_size = (3,1),padding = 'Same', 
#                 activation ='relu'))
#model.add(MaxPool2D(pool_size=(2,2), strides=(10,10)))
#model.add(Dropout(0.25))
# fully connected
model.add(Flatten())
model.add(Dense(800, activation = "relu"))
model.add(Dropout(0.25))
model.add(Dense(400, activation = "relu"))
model.add(Dropout(0.25))
model.add(Dense(100, activation = "relu"))
#Define Optimizer
optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
# Compile the model
model.compile(optimizer = optimizer , loss = "mean_squared_error", metrics=["accuracy"])
epochs = 20  # for better result increase the epochs
batch_size = 200
# data augmentation
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # dimesion reduction
        rotation_range=5,  # randomly rotate images in the range 5 degrees
        zoom_range = 0.1, # Randomly zoom image 10%
        width_shift_range=0.1,  # randomly shift images horizontally 10%
        height_shift_range=0.1,  # randomly shift images vertically 10%
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(X_train)
# Fit the model
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_test,Y_test), steps_per_epoch=X_train.shape[0] // batch_size)
# Plot the loss and accuracy curves for training and validation 
plt.plot(history.history['loss'], color='b', label="accuracy")
plt.title("validation loss")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
# confusion matrix
import seaborn as sns
# Predict the values from the validation dataset
Y_pred = model.predict(X_test)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_test,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
print(Y_pred)
print(Y_test)
error=np.abs((Y_pred-Y_test))/Y_test
from numpy import array  
from numpy.linalg import norm 
l1 = norm(np.abs((Y_pred-Y_test)),2)/norm(Y_test,2)*100
print(error)
print(l1)
angles = np.arange(0, 360, 3.6)

print(angles)
x1 = [m * np.cos(np.radians(a)) for m, a in zip(Y_pred[15,:], angles)]
y1 = [m * np.sin(np.radians(a)) for m, a in zip(Y_pred[15,:], angles)]
# Create a polar plot

fig, ax1 = plt.subplots(subplot_kw=dict(projection='polar'))
ax1.plot(np.radians(angles), Y_pred[15,:], 'o',color='red')
ax1.set_title('predicted shape')
# Show the plot
x2 = [m * np.cos(np.radians(a)) for m, a in zip(Y_test[15,:], angles)]
y2 = [m * np.sin(np.radians(a)) for m, a in zip(Y_test[15,:], angles)]
# Create a polar plot
fig, ax2 = plt.subplots(subplot_kw=dict(projection='polar'))
ax2.plot(np.radians(angles), Y_test[15,:], 'o')
ax2.set_title('real shape')
# Show the plot
plt.show()
x1 = [m * np.cos(np.radians(a)) for m, a in zip(Y_pred[15,:], angles)]
y1 = [m * np.sin(np.radians(a)) for m, a in zip(Y_pred[15,:], angles)]
x2 = [m * np.cos(np.radians(a)) for m, a in zip(Y_test[15,:], angles)]
y2 = [m * np.sin(np.radians(a)) for m, a in zip(Y_test[15,:], angles)]

fig, ax = plt.subplots()
# Plot the first plot
ax.plot(x1, y1,color='red')

# Plot the second plot
ax.plot(x2, y2)

# Show the figure
plt.show()