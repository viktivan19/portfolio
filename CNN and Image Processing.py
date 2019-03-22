import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping


# Read in images
# Load the image
data = plt.imread('bricks.png')

# Display the image
plt.imshow(data)
plt.show()


#Dimensions of an image
# the image has three dimensions: the first index can be between int(0-1200), the second can be between int(0-2400) and the third between float(0-1): RedGreenBlue
data[1200, 2400, 3]

# Set the red channel in this part of the image to 1 (the 1st of the 3 is the Red value)
data[:10,:10, 0] = 1

# Set the green channel in this part of the image to 0 (the second of the 3 is the Green value)
data[:10,:10, 1] = 0

# Set the blue channel in this part of the image to 0 (the third
data[:10,:10, 2] = 0

# Visualize the result
plt.imshow(data)
plt.show()

#Hot Encoding
labels = ['shoe', 'shirt', 'shoe', 'shirt', 'dress', 'dress', 'dress']

# The number of image categories
n_categories = 3

# The unique values of categories in the data
categories = np.array(["shirt", "dress", "shoe"])

# Initialize ohe_labels as all zeros
ohe_labels = np.zeros((len(labels), n_categories))

# Loop over the labels
for ii in range(len(labels)):
    # Find the location of this label in the categories variable
    # this is the rows
    jj = np.where(categories == labels[ii])
    # Set the corresponding zero to one (columns, rows)
    ohe_labels[ii, jj] = 1



#input_shape: pixels*pixels


model = Sequential()
model.add(Dense(50, activation='relu', input_shape = (n_cols,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
# Verify that model contains information from compiling
print("Loss function: " + model.loss)
#Validation
# Fit model_1
model_1_training = model_1.fit(predictors, target, epochs=15, validation_split=0.2, callbacks=[early_stopping_monitor], verbose=False)

# Fit model_2
model_2_training = model_2.fit(predictors, target, epochs=15, validation_split=0.2, callbacks=[early_stopping_monitor], verbose=False)

# Create the plot
plt.plot(model_1_training.history['val_loss'], 'r', model_2_training.history['val_loss'], 'b')
plt.xlabel('Epochs')
plt.ylabel('Validation score')
plt.show()

