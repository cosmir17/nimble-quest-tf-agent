import cv2
import numpy as np
import pytesseract as pt
import tensorflow as tf
from PIL import Image
from mss import mss
from tensorflow.keras.models import load_model

from screen_classifier.game_stage import GameStage

weight_path = "screen_classifier/nq_screen_weight.h5"
loaded_model = load_model(weight_path)


# 1261 X 702
def capture_window():
    with mss() as sct:
        monitor = {"top": 30, "left": 40, "width": 1261, "height": 702}
        sct_img = sct.grab(monitor)
        img = Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX')
        img = np.array(img)
        img = img / 255.0
        return img.astype(np.float32)


def which_stage(img):
    img_resized = tf.image.resize(img, (100, 100))
    prediction = loaded_model.predict(np.array([img_resized]))
    found = np.argmax(prediction)
    return GameStage(found)


def capture_window_raw():
    with mss() as sct:
        monitor = {"top": 30, "left": 40, "width": 1261, "height": 702}
        sct_img = sct.grab(monitor)
        return sct_img


def convert_raw_scren_to_tf_np(sct_img):
    img = Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX')
    img = np.array(img)
    img = img / 255.0
    return img.astype(np.float32)


def extract_jewel_reward_game_in_progress(np_img):
    w, h = 140, 50
    cropped = tf.image.crop_to_bounding_box(np_img, 40, 1120, h, w)
    img = tf.keras.preprocessing.image.array_to_img(cropped)
    jewel_reward = pt.image_to_string(img, config='--psm 6 --oem 3')
    return jewel_reward


# "width": 1261, "height": 702
def extract_kill_reward_game_over(np_img):
    w, h = 170, 50
    cropped = tf.image.crop_to_bounding_box(np_img, 282, 460, h, w)
    return recogonise_digit_image(cropped)


def extract_jewel_reward_game_over(np_img):
    w, h = 140, 35
    cropped = tf.image.crop_to_bounding_box(np_img, 349, 427, h, w)
    return recogonise_digit_image(cropped)


def extract_coin_reward_game_over(np_img):
    w, h = 100, 45
    cropped = tf.image.crop_to_bounding_box(np_img, 30, 635, h, w)
    return recogonise_digit_image(cropped)


def recogonise_digit_image(cropped):
    img = tf.keras.preprocessing.image.array_to_img(cropped)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img, 3)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.bilateralFilter(img, 9, 75, 75)
    (thresh, img_bw) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_bw = cv2.bitwise_not(img_bw)

    multiple_digits_config = '--oem 3 --psm 6 outputbase digits'
    single_digit_config = '--psm 13 --oem 3 outputbase digits -c tessedit_char_whitelist=0123456789'

    digit_string = pt.image_to_string(img_bw, config=multiple_digits_config)
    digit_string = digit_string.replace("\n", "")
    digit_string = digit_string.replace("\f", "")

    if not digit_string.isnumeric() and digit_string != "":
        print("digit_string contains non-numeric data : " + digit_string)
        digit_string = pt.image_to_string(img, config=multiple_digits_config)
    try:
        digits = int(digit_string)
        return digits
    except:
        return 0
