import cv2
import numpy as np
import pytesseract as pt
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from mss import mss
from tensorflow.keras.models import load_model
from imutils import grab_contours, contours
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


def extract_kill_reward_game_over(np_img):
    w, h = 170, 50
    cropped = tf.image.crop_to_bounding_box(np_img, 282, 460, h, w)
    return recogonise_digit_image(cropped)


def extract_kill_game_in_progress(np_img):
    w, h = 160, 60
    cropped = tf.image.crop_to_bounding_box(np_img, 30, 1090, h, w)
    return recogonise_digit_image(cropped)


def extract_jewel_game_in_progress(np_img):
    w, h = 145, 45
    cropped = tf.image.crop_to_bounding_box(np_img, 100, 1110, h, w)
    return recogonise_digit_image(cropped, True, True)


def extract_jewel_reward_game_over(np_img):
    w, h = 140, 35
    cropped = tf.image.crop_to_bounding_box(np_img, 349, 427, h, w)
    return recogonise_digit_image(cropped)


def extract_coin_reward_game_over(np_img):
    w, h = 100, 45
    cropped = tf.image.crop_to_bounding_box(np_img, 30, 635, h, w)
    return recogonise_digit_image(cropped)


#issue no.1 pytesseract doesn't recognise nq digits
#issue no.2 opencv doesn't extract digits well when the bckground is not obvious
def recogonise_digit_image(cropped, remove_first_obj=False, background_susceptible=False):
    img = tf.keras.preprocessing.image.array_to_img(cropped)
    mask = np.zeros(cropped.shape, dtype=np.float32)
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    (cnts, _) = contours.sort_contours(cnts, method="left-to-right")
    concatenated_img = None
    roi_number = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 800 and area > 200:
            x, y, w, h = cv2.boundingRect(c)
            roi = 255 - thresh[y:y + h, x:x + w]
            cv2.drawContours(mask, [c], -1, (255, 255, 255), -1)
            roi_height, _ = np.array(roi).shape
            gap_img = np.zeros((roi_height, 5), dtype=np.uint8)
            if remove_first_obj and concatenated_img is None:
                concatenated_img = np.zeros((roi_height, 100), dtype=np.uint8)
                continue
            if concatenated_img is None:
                concatenated_img = np.zeros((roi_height, 100), dtype=np.uint8)
            concatenated_img = hconcat_resize_min([concatenated_img, roi, gap_img])
            roi_number += 1
    roi_height, _ = concatenated_img.shape
    concatenated_img_2 = np.zeros((roi_height, 100), dtype=np.uint8)

    img = cv2.hconcat([concatenated_img, concatenated_img_2])
    h, w = img.shape
    concatenated_img_3 = np.zeros((h, w), dtype=np.uint8)
    img = cv2.vconcat([img, concatenated_img_3])
    img = cv2.vconcat([concatenated_img_3, img])

    img = cv2.bilateralFilter(img, 9, 75, 75)
    img_bw = cv2.bitwise_not(img)

    cv2.imshow('thresh', img_bw)
    cv2.waitKey(0)
    multiple_digits_config = '--oem 3 --psm 6 outputbase digits'
    single_digit_config = '--psm 13 --oem 3 outputbase digits -c tessedit_char_whitelist=0123456789'

    digit_string = digit_image_string(img_bw, multiple_digits_config)
    if not digit_string.isnumeric() and digit_string != "":
        print("digit_string contains non-numeric data : " + digit_string)
        digit_string = digit_image_string(img_bw, single_digit_config)
    try:
        digits = int(digit_string)
        return digits
    except:
        return 0


def digit_image_string(img_bw, multiple_digits_config):
    digit_string = pt.image_to_string(img_bw, config=multiple_digits_config)
    print("recognised string: " + digit_string)
    digit_string = digit_string.replace("\n", "")
    digit_string = digit_string.replace("\f", "")
    return digit_string


def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)


def is_back_button_selected(np_img):
    back_button_color = np_img[585][700]
    # [0.07450981 0.60784316 0.00392157] highlighted
    # [0.19607843 0.6117647  0.14509805] unhighlighted
    if np.allclose(back_button_color, np.array([0.07450981, 0.60784316, 0.00392157])):
        return True
    else:
        return False
