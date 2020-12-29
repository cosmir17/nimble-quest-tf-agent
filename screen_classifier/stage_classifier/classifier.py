import h5py
import os
import shutil
import datetime
import platform
from sklearn.metrics import confusion_matrix
from confusion_matrix_plotter import *
from tensorflow import keras
from tensorflow.keras import utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.layers import Input, Conv2D, Flatten, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential, load_model

# import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     tf.config.experimental.set_virtual_device_configuration(gpus[0],
#             [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6000)])
#     print("****************** setting memory limit 5000mb ******************")
#   except RuntimeError as e:
#     print(e)
# debug_path = "logs\\debug\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tf.debugging.experimental.enable_dump_debug_info(debug_path, tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)

shutil.rmtree('logs')

train_image_folder = "screenshots"
test_image_folder = "test_screenshots"

class_names = ['in_progress', 'game_over', 'starting_page', 'interval',
               'character_upgrade', 'interval_sorry', 'interval_upgrade',
                'paused_in_progress', 'game_over_sorry',
                'store_page', 'main_page', 'died']

weightPath = "nq_screen_weight.h5"
pb_path = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

cm_file_writer_cm = tf.summary.create_file_writer(pb_path + '/cm')


def load_images_to_np_array(path_array, image_folder):
    image_array = []
    for path in path_array:
        img = tf.image.decode_png(tf.io.read_file(image_folder + "/" + path), channels=3)
        img = tf.cast(img, tf.float32) / 255.0
        arr = keras.preprocessing.image.img_to_array(img)
        arr = tf.image.resize(arr, (100, 100))
        # arr = tf.image.resize(arr, (80, 80))
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


os_name = platform.system()

train_image_file_list = os.listdir(train_image_folder)
if os_name == "Darwin":
    train_image_file_list.remove(".DS_Store")
x_train, y_train = convert_to_x_y(train_image_file_list, train_image_folder)

test_image_file_list = os.listdir(test_image_folder)
if os_name == "Darwin":
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
nq_network = Dropout(0.5)(nq_network)
nq_network = Dense(12, activation='softmax')(nq_network)

model = Model(inputs=i, outputs=nq_network)
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0004), loss='categorical_crossentropy', metrics='accuracy')

print("x_train[0].shape:", x_train[0].shape)


def log_confusion_matrix(epoch, logs):
    p_train = model.predict(x_train).argmax(axis=1)  # side effect, necessary evil due to the tf design
    y_train_m = y_train.argmax(axis=1)
    cm_train = confusion_matrix(y_train_m, p_train)
    train_figure = plot_confusion_matrix(cm_train, list(range(12)))
    cm_train_image = plot_figure_to_image(train_figure)
    mismatched_train_img = plot_mismatched_images(p_train, x_train, y_train_m, class_names)

    p_test_m = model.predict(x_test).argmax(axis=1)  # side effect, necessary evil due to the tf design
    y_test_m = y_test.argmax(axis=1)
    cm_test = confusion_matrix(y_test_m, p_test_m)
    test_figure = plot_confusion_matrix(cm_test, list(range(12)))
    cm_test_image = plot_figure_to_image(test_figure)
    mismatched_test_img = plot_mismatched_images(p_test_m, x_test, y_test_m, class_names)

    with cm_file_writer_cm.as_default():
        tf.summary.image("Confusion Matrix: cm_train_image", cm_train_image, step=epoch)
        tf.summary.image("Confusion Matrix: cm_test_image", cm_test_image, step=epoch)
        tf.summary.image("Mismatched: train_images", mismatched_train_img, step=epoch)
        tf.summary.image("Mismatched: test_images", mismatched_test_img, step=epoch)


checkpoint = ModelCheckpoint(weightPath, monitor='loss', verbose=1, save_best_only=True, mode='min')
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=pb_path, histogram_freq=1)
cm_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)
callbacks_list = [checkpoint, tensorboard_callback, cm_callback]
# tensorboard --logdir logs/fit
batch_size = 100


data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        #brightness_range=[0.9,1.1], this doesn't work and corrupt the training process
        horizontal_flip=True, vertical_flip=True,
        width_shift_range=0.1, height_shift_range=0.1)
train_generator = data_generator.flow(x_train, y_train, batch_size)

############# LOAD ################### LOAD ################### LOAD ######
model = load_model(weightPath)
model.summary()

r = model.fit(train_generator, validation_data=(x_test, y_test), epochs=1000, batch_size=batch_size, callbacks=callbacks_list)

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
# r = model.fit(train_generator, validation_data=(x_test, y_test), epochs=3000, batch_size=batch_size, callbacks=callbacks_list)
############### Training ############### Training ############### Training ###############
