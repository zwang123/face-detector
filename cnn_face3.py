#Import TensorFlow
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt

from image_reader import load_data_from_folder
from image_reader import load_scenery_from_folder
from image_reader import load_zoom
from image_reader import sample_images

import numpy as np

img, lbl = load_data_from_folder(saved=True)
lbl = lbl[:, :1]
# load data from UTK, 7:1 as train/test ratio 
(train_images, train_labels), (test_images, test_labels) = \
sample_images(img, lbl, ratio=7)

# load data from random scenery pictures, 1:1 as train/test ratio 
(tr_images, tr_labels), (ts_images, ts_labels) = \
sample_images(*load_scenery_from_folder(saved=True) , ratio=1)

train_images = np.append(train_images, tr_images, axis=0)
train_labels = np.append(train_labels, tr_labels, axis=0)
ts_images = np.append(test_images, ts_images, axis=0)
ts_labels = np.append(test_labels, ts_labels, axis=0)

# load data from an independent dataset as test case
img, lbl = load_data_from_folder(saved=True, data_folder='data/Image/*/',
        load_fxn=load_zoom)
test_images = np.append(img, ts_images, axis=0)
test_labels = np.append(lbl, ts_labels, axis=0)

print("train images", train_images.shape)
print("train labels", train_labels.shape)
print("test images", test_images.shape)
print("test labels", test_labels.shape)

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

##Verify the data
##class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
##               'dog', 'frog', 'horse', 'ship', 'truck']
#class_names = ['nonface', 'face']
#
#plt.figure(figsize=(10,10))
#for i in range(25):
#    plt.subplot(5,5,i+1)
#    plt.xticks([])
#    plt.yticks([])
#    plt.grid(False)
#    plt.imshow(train_images[i], cmap=plt.cm.binary)
#    # The CIFAR labels happen to be arrays,
#    # which is why you need the extra index
#    plt.xlabel(class_names[train_labels[i][0]])
#plt.show()

#Create the convolutional base
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', 
    input_shape=(50, 50, 3),
    kernel_regularizer=regularizers.l2(0.0001)
    ))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu',
    kernel_regularizer=regularizers.l2(0.0001)
    ))
model.add(layers.MaxPooling2D((2, 2)))

#Add Dense layers on top
model.add(layers.Flatten())
model.add(layers.Dense(2, activation='softmax',
    kernel_regularizer=regularizers.l2(0.0001)
    ))

#Let's display the architecture of our model so far.
model.summary()

#Compile and train the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=5, 
                    validation_data=(test_images, test_labels))

#Evaluate the model
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

#plt.show()

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print("test accuracy", test_acc)
