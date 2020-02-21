import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import keras
from time import time
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

image_width = 128
image_height = 128
image_channels = 3
image_sizes = (image_height, image_width)
input_shape = (image_height, image_width, image_channels)
num_classes = 6
df = pd.read_csv('./train_vision.csv')
# train image

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
# model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
# model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
# model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.summary()

earlystop = EarlyStopping(patience=7) #monitor='val_loss'
lr_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.4, min_lr=0.00005)

tensorboard_cb = keras.callbacks.TensorBoard(log_dir="E:/ML02/kaggle_img/logs/{}".format(time()))
callbacks = [earlystop, tensorboard_cb, lr_reduction]

# data prepare
df['label'] = df['label'].replace({1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6'})
train_df, validate_df = train_test_split(df, test_size=0.2, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)

train_df['label'].value_counts().plot.bar()
validate_df['label'].value_counts().plot.bar()

total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
batch_size = 64

# Data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    width_shift_range=0.01,
    height_shift_range=0.01
)
train_generator = train_datagen.flow_from_dataframe(train_df, './faces_images', x_col='filename', y_col='label',
                                                    target_size=image_sizes, class_mode='categorical',
                                                    batch_size=batch_size)

validation_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = validation_datagen.flow_from_dataframe(validate_df, './faces_images', x_col='filename',
                                                              y_col='label', target_size=image_sizes,
                                                              class_mode='categorical',
                                                              batch_size=batch_size)
# fit model
epochs = 25  # 15
history = model.fit_generator(train_generator, epochs=epochs,
                              validation_data=validation_generator,
                              validation_steps=total_validate // batch_size,
                              steps_per_epoch=total_train // batch_size,
                              callbacks=callbacks)
                              # shuffle=True)

# test data
test_df = pd.read_csv('./test_vision.csv')
nb_samples = test_df.shape[0]

# test data generatoring...
test_gen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_gen.flow_from_dataframe(test_df, './faces_images', x_col='filename', y_col=None,
                                              class_mode=None, target_size=image_sizes,
                                              batch_size=batch_size, shuffle=False)

predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples / batch_size))

test_df['label'] = np.argmax(predict, axis=-1)

label_map = dict((v, k) for k, v in train_generator.class_indices.items())
test_df['label'] = test_df['label'].replace(label_map)
test_df['label'] = test_df['label'].replace({'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6})

# test_df['label'].value_counts().plot.bar()

submission = test_df.copy()
submission.rename(columns={'label': 'prediction'}, inplace=True)
submission.drop(['filename'], axis=1, inplace=True)
submission.set_index('prediction')
submission.to_csv('submission-{}.csv'.format(time()), index=False)
