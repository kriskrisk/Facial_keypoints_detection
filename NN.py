import os

import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

import matplotlib.pyplot as plt


FTRAIN = '~/Documents/faceRecognition/training.csv'
FTEST = '~/Documents/faceRecognition/test.csv'


def load(test=False):
    fname = FTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(fname))  # load pandas dataframe

    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    df = df.dropna()  # drop all rows that have missing values in them

    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)

    if not test:  # only FTRAIN has any target columns
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48  # scale target coordinates to [-1, 1]
        X, y = shuffle(X, y, random_state=42)  # shuffle train data
        y = y.astype(np.float32)
    else:
        y = None

    return X, y

def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)

# some settings
#batch_size = 32
num_classes = 30
nepochs = 400
#data_augmentation = True
#num_predictions = 30

X, y = load()

# create model
model = Sequential()
model.add(Dense(100, input_dim=9216, activation='relu'))
model.add(Dense(num_classes, activation='relu'))

# initiate RMSprop optimizer
opt = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=True)

# compile model
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])

# fit the model
history = model.fit(X, y, epochs=nepochs, batch_size=1, validation_split=0.33)
print(history.history.keys())

#pyplot.plot(train_loss, linewidth=3, label="train")
#pyplot.plot(valid_loss, linewidth=3, label="valid")
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.grid()
plt.legend(['train', 'test'], loc='upper left')
plt.title('model loss')
plt.xlabel("epoch")
plt.ylabel("loss")
plt.ylim(1e-3, 1e-2)
#pyplot.yscale("log")
plt.show()

