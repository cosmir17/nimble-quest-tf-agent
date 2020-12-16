import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.layers import Input, Conv2D, Flatten, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential, load_model

# 0 game in prgress
# 1 game over
# 2 starting page
# 3 interval

# 4 character upgrade
# 5 interval sorry page
# 6 interval upgrade page
# 7 pause game in progress
# 8 game end sorry
# 9 store page
# 10 main page


train_image_folder = "screenshots"
test_image_folder = "test_screenshots"


def load_images_to_np_array(path_array, image_folder):
    image_array = []
    for path in path_array:
        img = tf.image.decode_png(tf.io.read_file(image_folder + "/" + path), channels=3)
        img = tf.cast(img, tf.float32) / 255.0
        arr = keras.preprocessing.image.img_to_array(img)
        arr = tf.image.resize(arr, (100, 100))
        image_array.append(arr)
    return np.asarray(image_array)


def convert_to_x_y(image_file_list, image_folder):
    image_category_list = []
    for name in image_file_list:
        name_arr = name.split("_")
        image_category_list.append(name_arr[1])

    y = np.asarray(image_category_list)
    y = utils.to_categorical(y, 12)
    x = load_images_to_np_array(image_file_list, image_folder)
    return x, y


train_image_file_list = os.listdir(train_image_folder)
train_image_file_list.remove(".DS_Store")
x_train, y_train = convert_to_x_y(train_image_file_list, train_image_folder)

test_image_file_list = os.listdir(test_image_folder)
test_image_file_list.remove(".DS_Store")
x_test, y_test = convert_to_x_y(test_image_file_list, test_image_folder)

# strategy = tf.distribute.MirroredStrategy()
# with strategy.scope():
i = Input(shape=x_train[0].shape)
nq_network = Conv2D(90, (3, 3), activation='relu', padding='same')(i)
nq_network = BatchNormalization()(nq_network)
nq_network = MaxPooling2D((2, 2))(nq_network)
nq_network = Dropout(0.05)(nq_network)
nq_network = Conv2D(150, (3, 3), activation='relu', padding='same')(nq_network)
nq_network = BatchNormalization()(nq_network)
nq_network = MaxPooling2D((2, 2))(nq_network)
nq_network = Dropout(0.05)(nq_network)
nq_network = Conv2D(250, (4, 4), activation='relu', padding='same')(nq_network)
nq_network = BatchNormalization()(nq_network)
nq_network = MaxPooling2D((2, 2))(nq_network)
nq_network = Dropout(0.05)(nq_network)
nq_network = Conv2D(500, (4, 4), activation='relu', padding='same')(nq_network)
nq_network = BatchNormalization()(nq_network)
nq_network = MaxPooling2D((2, 2))(nq_network)
nq_network = Dropout(0.05)(nq_network)
nq_network = Flatten()(nq_network)
nq_network = Dropout(0.07)(nq_network)
nq_network = Dense(1000, activation='relu')(nq_network)
nq_network = Dropout(0.05)(nq_network)
nq_network = Dense(500, activation='relu')(nq_network)
nq_network = Dropout(0.05)(nq_network)
nq_network = Dense(500, activation='relu')(nq_network)
nq_network = Dropout(0.05)(nq_network)
nq_network = Dense(12, activation='softmax')(nq_network)

model = Model(inputs=i, outputs=nq_network)
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')

print("x_train[0].shape:", x_train[0].shape)

weightPath = "nq_screen_weight.h5"
checkpoint = ModelCheckpoint(weightPath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
batch_size = 200

data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        #brightness_range=[0.9,1.1], this doesn't work and corrupt the training process
        horizontal_flip=True, vertical_flip=True,
        width_shift_range=0.1, height_shift_range=0.1)
train_generator = data_generator.flow(x_train, y_train, batch_size)

############# LOAD ################### LOAD ################### LOAD ######
loaded_model = load_model(weightPath)
# loaded_model.summary()
#
r = loaded_model.fit(train_generator, validation_data=(x_test, y_test), epochs=100, batch_size=batch_size, callbacks=callbacks_list)

# prediction = tf.image.decode_png(tf.io.read_file("test_screenshot/113_8_.png"), channels=3)
# prediction = tf.cast(prediction, tf.float32) / 255.0
# prediction = tf.image.resize(prediction, (100, 100))
# prediction = keras.preprocessing.image.img_to_array(prediction)
# p = np.array([prediction])
# matrix = loaded_model.predict(p)
# print(matrix)
# print(np.argmax(matrix))
############# LOAD ################### LOAD ################### LOAD ######

############### Training ############### Training ############### Training ###############
# batch_size = 200
# steps_per_epoch = x_train.shape[0]
# r = model.fit(train_generator, validation_data=(x_test, y_test), epochs=6000, batch_size=batch_size, callbacks=callbacks_list)
############### Training ############### Training ############### Training ###############

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# Plot accuracy per iteration
plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()

# Confusion matrix part is copied from: https://colab.research.google.com/drive/1pdzZ2MB2g6CT_-bT0D0bO2IKyghOhlM_
from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
  """
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
  """
  if normalize:
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      print("Normalized confusion matrix")
  else:
      print('Confusion matrix, without normalization')

  print(cm)

  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j], fmt),
               horizontalalignment="center",
               color="white" if cm[i, j] > thresh else "black")

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show()


# p_test = loaded_model.predict(x_train).argmax(axis=1)
# y_train = y_train.argmax(axis=1)
# cm = confusion_matrix(y_train, p_test)
# plot_confusion_matrix(cm, list(range(12)))
#
# misclassified_idx = np.where(p_test != y_train)[0]
# if misclassified_idx.size > 0:
#     i = np.random.choice(misclassified_idx)
#     plt.imshow(x_train[i], cmap='gray')
#     print("misclassified index : " + str(misclassified_idx))
#     plt.title("True label: %s Predicted: %s" % (y_train[i], p_test[i]))
#     plt.show()


p_test = loaded_model.predict(x_test).argmax(axis=1)
y_test = y_test.argmax(axis=1)
cm = confusion_matrix(y_test, p_test)
plot_confusion_matrix(cm, list(range(12)))

misclassified_idx = np.where(p_test != y_test)[0]
if misclassified_idx.size > 0:
    i = np.random.choice(misclassified_idx)
    plt.imshow(x_test[i], cmap='gray')
    print("misclassified index : " + str(misclassified_idx))
    plt.title("True label: %s Predicted: %s" % (y_test[i], p_test[i]))
    plt.show()


# # Confusion matrix, without normalization
# [[85  0  0  0  0  0  0  0  0  0  0  0]
#  [ 0 39  0  0  0  0  0  0  0  0  0  0]
#  [ 0  0 12  0  0  0  0  0  0  0  0  0]
#  [ 0  0  0 18  0  0  0  0  0  0  0  0]
#  [ 0  0  0  0  3  0  0  0  0  0  0  0]
#  [ 0  0  0  0  0  8  0  0  0  0  0  0]
#  [ 0  0  0  0  0  0 10  0  0  0  0  0]
#  [ 0  0  0  0  0  0  0  9  0  0  0  0]
#  [ 0  0  0  0  0  0  0  0  4  0  0  0]
#  [ 0  0  0  0  0  0  0  0  0  6  0  0]
#  [ 0  0  0  0  0  0  0  0  0  0  1  0]
#  [ 0  0  0  0  0  0  0  0  0  0  0 70]]
