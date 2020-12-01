import cv2
import numpy as np
import pytesseract as pt
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from mss import mss
from tensorflow.keras.models import load_model
from imutils import grab_contours, contours
from screen_classifier.stage_classifier.game_stage import GameStage

stage_weight_path = "screen_classifier/stage_classifier/nq_screen_weight.h5"
score_weight_path = "screen_classifier/score_classifier/nq_score_weight.h5"
stage_model = load_model(stage_weight_path)
score_model = load_model(score_weight_path)

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
    prediction = stage_model.predict(np.array([img_resized]))
    found = np.argmax(prediction)
    return GameStage(found)


def predict_from_cnn(img):
    img = tf.image.resize(img, (35, 35))
    if np.mean(img) == 255 or np.mean(img) == 0:
        found = ""
    else:
        img = np.array(img) / 255.0
        prediction = score_model.predict(np.array([img]))
        found = np.argmax(prediction)
        found = str(found)
        if found == "10":
            found = ""
    return found

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


def extract_kill_reward_game_over(np_img):
    w, h = 170, 50
    cropped = tf.image.crop_to_bounding_box(np_img, 282, 460, h, w)
    return recognise_digit_image(cropped)


def extract_kill_game_in_progress(np_img):
    w, h = 160, 60
    cropped = tf.image.crop_to_bounding_box(np_img, 30, 1090, h, w)
    return recognise_digit_image(cropped)


def extract_jewel_game_in_progress(np_img):
    w, h = 145, 45
    cropped = tf.image.crop_to_bounding_box(np_img, 100, 1110, h, w)
    color_range = [(36, 25, 25), (86, 255, 255)]
    # color_range = [(36, 25, 25), (86, 255, 255)]
    return recognise_digit_image(cropped, color_range)


def extract_jewel_reward_game_over(np_img):
    w, h = 140, 35
    cropped = tf.image.crop_to_bounding_box(np_img, 349, 427, h, w)
    return recognise_digit_image(cropped)


def extract_coin_reward_game_over(np_img):
    w, h = 100, 45
    cropped = tf.image.crop_to_bounding_box(np_img, 30, 635, h, w)
    return recognise_digit_image(cropped)


def recognise_digit_image(cropped, color_range=None):
    img = tf.keras.preprocessing.image.array_to_img(cropped)
    if color_range is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2HSV)
        img = cv2.inRange(img, color_range[0], color_range[1])
        thresh = cv2.bitwise_not(img)
    else:
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    mask = np.zeros(thresh.shape, dtype=np.float32)
    cnts = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    try:
        if len(cnts) == 2:
            cnts = cnts[0]
        else:
            cnts = cnts[1]
        (cnts, _) = contours.sort_contours(cnts, method="left-to-right")
        score = ""
        for c in cnts:
            area = cv2.contourArea(c)
            if color_range is None:
                threshold = 500
            else:
                threshold = 300
            if area < 800 and area > threshold:
                x, y, w, h = cv2.boundingRect(c)
                roi = 255 - thresh[y:y + h, x:x + w]
                cv2.drawContours(mask, [c], -1, (255, 255, 255), -1)
                roi = cv2.bitwise_not(roi)
                roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
                score = score + predict_from_cnn(roi)
        score_int = convert_score_to_int(score)
        return score_int
    except:
        print("score recognition, contour error, returning score 0")
        return None


def convert_score_to_int(score):
    score = score.strip()
    if score == "":
        score = None
    else:
        score = int(score)
    return score


def is_back_button_selected(np_img):
    back_button_color = np_img[585][700]
    # [0.07450981 0.60784316 0.00392157] highlighted
    # [0.19607843 0.6117647  0.14509805] unhighlighted
    if np.allclose(back_button_color, np.array([0.07450981, 0.60784316, 0.00392157])):
        return True
    else:
        return False



# prediction = tf.image.decode_png(tf.io.read_file("83_0_.png"), channels=3)
# prediction = tf.keras.preprocessing.image.img_to_array(prediction)
# # selected = is_back_button_selected(prediction)
# selected = extract_kill_game_in_progress(prediction)
# print("score: " + str(selected))
