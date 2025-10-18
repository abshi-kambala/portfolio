# make sure tensorflow_datasets package is at newest possible version
# to ensure that dataset can be loaded properly.
# runtime will need to be restarted after this package is upgraded.
# press the "RESTART RUNTIME" button that appears,
# and DO NOT RUN THIS CELL after the runtime has restarted.
# !pip install --upgrade tensorflow_datasets


import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image




# verify that the tensorflow_datasets package has been updated
print(tfds.__version__)
# version 4.6.0 has been verified to work for loading the datasets


ds_train, train_info = tfds.load('malaria', split='train', with_info=True, as_supervised=True)




# function to standardize the image size to what the model expects
def resize_image(image_tensor: tf.Tensor, label_tensor: tf.Tensor):
  im = image.array_to_img(image_tensor.numpy())
  im = im.resize((128, 128))
  im = image.img_to_array(im)
  image_tensor = tf.convert_to_tensor(im, dtype=tf.uint8)
  return image_tensor, label_tensor




seed = 51
tf.random.set_seed(seed)
ds_train = ds_train.map(lambda x,y: tf.py_function(func=resize_image, inp=[x,y], Tout=(tf.uint8, tf.int64)))
ds_train.shuffle(buffer_size=1024, seed=seed)




# this code runs for a while
# go get a cup of coffee!
images = []
labels = []
for image_0, label in ds_train:
  images.append(image_0.numpy())
  labels.append(label.numpy())
  if len(images) > 5000:
    break

from sklearn.model_selection import train_test_split

# Convert to numpy arrays
images_np = np.array(images)
labels_np = np.array(labels)

# Split the data
X_train, X_valid, y_train, y_valid = train_test_split(
    images_np, labels_np, random_state=0, test_size=0.2
)

# Normalize
X_train = X_train.astype('float32') / 255.0
X_valid = X_valid.astype('float32') / 255.0

import logging
logging.getLogger('tensorflow').disabled = True

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, BatchNormalization
from tensorflow.keras import regularizers

base_model = MobileNetV2(
    input_shape=(128, 128, 3),
    include_top=False,
    weights='imagenet'
)


base_model.trainable = False


inputs = Input(shape=(128, 128, 3))


x = tf.keras.layers.RandomFlip("horizontal")(inputs)
x = tf.keras.layers.RandomRotation(0.1)(x)


x = base_model(x, training=False)


x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
outputs = Dense(2, activation='softmax')(x)


model = Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
history = model.fit(
    X_train, y_train,
    batch_size=128,
    epochs=10,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    ],
    validation_data=(X_valid, y_valid),
    verbose=1
)

history

'''

predicted_classes = model.predict(X_valid)
predicted_classes = np.argmax(predicted_classes, axis=1)
incorrect = np.nonzero(predicted_classes!=y_valid)[0]
class_names = ['infected', 'non-infected']

# Display the first 16 incorrectly classified images from the test data set
plt.figure(figsize=(15, 8))
for j, incorrect in enumerate(incorrect[0:8]):
    plt.subplot(2, 4, j+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(X_valid[incorrect].reshape(128, 128, 3), cmap="Reds")
    plt.title("Predicted: {}".format(class_names[predicted_classes[incorrect]]))
    plt.xlabel("Actual: {}".format(class_names[y_valid[incorrect]]))

'''

'''

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1]) # recall accuracy is between 0 to 1
plt.legend(loc='lower right') # specify location of the legend

test_loss, test_acc = model.evaluate(X_valid, y_valid, verbose=2)

'''

'''

images_np = np.array(images)
labels_np = np.array(labels)
which_class = 1  # for example: 1 for 'Parasitized', 0 for 'Uninfected'
training_images_class = images_np[labels_np == which_class]
fig = plt.figure(figsize=(12, 8))
columns = 5
rows = 3

for i in range(1, columns * rows + 1):
    if i >= len(training_images_class):
        break  # Avoid indexing beyond available images
    img = training_images_class[i]
    fig.add_subplot(rows, columns, i)
    plt.title("Label: " + str(which_class))
    plt.imshow(img.astype("uint8"))
    plt.axis('off')
plt.tight_layout()
plt.show()


'''
