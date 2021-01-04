import numpy as np
import platform
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from mss import mss
from tensorflow import keras
from tensorflow.keras.models import load_model
from screen_classifier.stage_classifier.game_stage import GameStage
import random
import cv2

stage_weight_path = "screen_classifier/stage_classifier/nq_screen_weight.h5"
stage_model = load_model(stage_weight_path)
os_name = platform.system()


# 1261 X 702
def capture_window():
    with mss() as sct:
        monitor = {"top": 30, "left": 40, "width": 1261, "height": 702}  # using Magnet on Mac
        if os_name == "Linux" or os_name == "Windows":
            monitor = {"top": 15, "left": 1, "width": 1261, "height": 702}
        sct_img = sct.grab(monitor)
        img = Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX')
        img = keras.preprocessing.image.img_to_array(img)
        img = tf.image.resize(img, (130, 130))
        is_image_black(img)
        return img


def which_stage(img):
    img_resized = tf.image.resize(img, (100, 100))  # stage
    img_resized = keras.preprocessing.image.img_to_array(img_resized)
    img_resized = img_resized / 255.0
    img_resized = img_resized.astype(np.float32)
    prediction = stage_model.predict_on_batch(np.array([img_resized]))
    found = np.argmax(prediction)
    return GameStage(found)


def capture_display():
    with mss() as sct:
        num = str(random.randint(0, 9434734734))
        filename = "screen_is_black_" + num + ".png"
        sct.shot(output=filename)
        return filename


def convert_raw_scren_to_tf_np(sct_img):
    img = Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX')
    img = np.array(img)
    img = img / 255.0
    return img.astype(np.float32)


def is_back_button_selected(np_img):
    np_img = np_img.numpy()
    back_button_color = np_img[103][71]
    if np.allclose(back_button_color, np.array([11, 83, 1]),
                   rtol=1.e-1, atol=1.e-1):
        return True
    else:
        return False


def is_image_black(image):
    gray = cv2.cvtColor(np.float32(image), cv2.COLOR_BGR2GRAY)
    if cv2.countNonZero(gray) == 0:
        # filename = capture_display()
        print("**screen_is_black : ")
