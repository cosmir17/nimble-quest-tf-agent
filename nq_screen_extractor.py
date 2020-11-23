from mss import mss
from PIL import Image
import tensorflow as tf
import cv2
from tensorflow import keras
import pytesseract as pt
import numpy as np
from PIL import Image,ImageOps
from tensorflow.keras.models import Sequential, load_model
from screen_classifier.game_stage import GameStage
import matplotlib.pyplot as plt

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


# b = capture_window()
# output3 = extract_coin_reward_game_over(b)
# output = extract_kill_reward_game_over(b)
# output2 = extract_jewel_reward_game_over(b)
# print(str(output) + "  ::::  " + str(output2) + ":::: " + str(output3))


# def convert_to_np_array(img):
#     arr = keras.preprocessing.image.img_to_array(img)
#     return arr
#
#
# def extract_string(img):
#     img = img.convert("L")
#     img = ImageOps.invert(img)
#     # img.show()
#     threshold = 240
#     table = []
#     pixel_array = img.load()
#     for y in range(img.size[1]):  # binaryzate it
#         List = []
#         for x in range(img.size[0]):
#             if pixel_array[x, y] < threshold:
#                 List.append(0)
#             else:
#                 List.append(255)
#         table.append(List)
#
#     img = Image.fromarray(np.array(img, dtype="uint8"))  # load the image from array.
#     img.show()
#     a = pt.image_to_string(img)
#     print(a)
#     return a

    # cv2.imshow('image', img)
    # cv2.waitKey(0)

    # cropped = tf.image.crop_to_bounding_box(np_img, 40, 1120, h, w)
    # img = tf.keras.preprocessing.image.array_to_img(cropped)
    # gray = cv2.medianBlur(img, 3)
    # image = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    # image = cv2.medianBlur(image, 5)
    # image = cv2.Canny(image, 100, 200)
    # kernel = np.ones((5, 5), np.uint8)
    # image = cv2.erode(image, kernel, iterations=1)
    # plt.imshow(image)
    # plt.show()

# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# # gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# gray = cv2.medianBlur(gray, 3)
# a = pt.image_to_string(gray)
# print(a)

# b = convert_to_np_array(b)
# c = extract_string(b)

# b = capture_window()
# plt.imshow(b)
# plt.show()
# print(which_stage(b))

# b = convert_to_np_array(b)
# c = extract_string(b)

    # plt.imshow(cropped)
    # plt.show()
    # np_img = np.array(cropped, dtype=np.uint8)
    # cropped = np.flip(np_img[:, :, :3], 2)
    # i = Image.fromarray(cropped.astype('uint8'), 'RGB')
    # plt.imshow(cropped)
    # plt.show()
    # data = np.zeros((h, w, 3), dtype=np.uint8)
    # data[0:256, 0:256] = [255, 0, 0]  # red patch in upper left
    # img = Image.fromarray(cropped)

# c = extract_string(np.float32(b))
# import pyscreenshot as ImageGrab
# img = ImageGrab.grab(bbox=(40, 30, 1300, 730))
# arr = keras.preprocessing.image.img_to_array(img)
# return arr
# # img = tf.image.decode_image(img, channels=3)
# # img = tf.cast(img, tf.float32) / 255.0
# # img = img / 255.0
# # img.show