# Imports
import tensorflow as tf # tested with 1.14.0
import numpy as np # tested with 1.16.4
import matplotlib.pyplot as plt #tested with 3.0.3
from sklearn.metrics import classification_report # tested with 0.21.2l

# deprecation warnings
import logging
logging.getLogger('tensorflow').disabled = True

# Fetch Fashion MNIST data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize / rescale these values.
x_train = x_train / 255.0
x_test = x_test / 255.0

# Map for human readable class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import regularizers

model = Sequential()
weight_decay = 0.0005


# Regulizer, Dropout layers, MaxPooling layers
model.add(Conv2D(64, (3, 3), padding='same',
                    input_shape=(28,28,1),kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))

# Build the model
model.compile(
    loss=tf.keras.losses.sparse_categorical_crossentropy, # loss function
    optimizer=tf.keras.optimizers.Adam(), # optimizer function
    metrics=['accuracy'] # reporting metric
)

# Model structure summary
print(model.summary())

# tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

# empty color dimension
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Train the CNN
history = model.fit(

      # Training data : features (images) and classes.
      x_train, y_train,

      # number of samples to work through before backpropagation
      batch_size=256,

      epochs=10,

      # validation split (20% of data for testing)
      validation_split=0.2,

      verbose=1)

'''
%tensorflow_version 2.x
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))
'''

'''
def gpu():
    with tf.device('/device:GPU:0'):
        # Train the CNN on the training data
        history = model.fit(
            x_train, y_train,
            batch_size=256,
            epochs=10,
            validation_split=0.2,
            verbose=1)
        return history
        
history = gpu()


import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1]) # recall accuracy is between 0 to 1
plt.legend(loc='lower right') # specify location of the legend

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

predicted_classes = model.predict(x_test)
predicted_classes = np.argmax(predicted_classes, axis=1)
incorrect = np.nonzero(predicted_classes!=y_test)[0]

# Display the first 16 incorrectly classified images from the test data set
plt.figure(figsize=(15, 8))
for j, incorrect in enumerate(incorrect[0:8]):
    plt.subplot(2, 4, j+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[incorrect].reshape(28, 28), cmap="Reds")
    plt.title("Predicted: {}".format(class_names[predicted_classes[incorrect]]))
    plt.xlabel("Actual: {}".format(class_names[y_test[incorrect]]))
'''

'''
# Getting the model without the last layers, trained with imagenet and with average pooling
K = tf.keras
base_model = K.applications.vgg16.VGG16(include_top=False,
    weights='imagenet',
    pooling='avg',
    input_shape=(32,32,3)
)

base_model = K.applications.resnet50.ResNet50(include_top=False,
    weights='imagenet',
    pooling='avg',
    input_shape=(32,32,3)
)
# VGG16 transfer learning
 
# create the new model applying the base_model (VGG16)
model= K.Sequential()
model.add(base_model)
model.add(K.layers.Flatten())
model.add(K.layers.Dense(2048, activation=('relu')))
model.add(K.layers.Dropout(0.3))
model.add(K.layers.Dense(1024, activation=('relu')))
model.add(K.layers.Dropout(0.3))
model.add(K.layers.Dense(512, activation=('relu')))
model.add(K.layers.Dropout(0.2))
model.add(K.layers.Dense(10, activation=('softmax')))

# Compiling model with adam
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# history = model.fit(
#             x=X_train, y=y_train,
#             batch_size=128,
#             validation_data=(X_val, y_val),
#             epochs=30,
#             verbose=1
#         )

# Reshape the input data to have 3 channels
x_train_resized = tf.image.resize(x_train, (32, 32))  # Resize the images from 28x28 to 32x32
x_train_resized = tf.image.grayscale_to_rgb(x_train_resized)  # Convert grayscale to RGB

# Compiling model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(
            x=x_train_resized, y=y_train,
            batch_size=128,
            validation_split=0.2,
            # validation_data=(X_val, y_val),
            epochs=30,
            verbose=1
        )

'''
