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


def read_score(img):
    img = tf.image.resize(img, (35, 35))
    # img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.array(img) / 255.0
    cv2.imshow('original_image', img)
    cv2.waitKey(0)
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
    return recognise_digit_image(cropped, True, color_range)


def extract_char_count_in_progress(np_img):
    w, h = 100, 45
    cropped = tf.image.crop_to_bounding_box(np_img, 30, 40, h, w)
    color_range = [(0, 0, 0), (0, 0, 0)]
    return recognise_digit_image(cropped, False, color_range)


def extract_jewel_reward_game_over(np_img):
    w, h = 140, 35
    cropped = tf.image.crop_to_bounding_box(np_img, 349, 427, h, w)
    return recognise_digit_image(cropped)


def extract_coin_reward_game_over(np_img):
    w, h = 100, 45
    cropped = tf.image.crop_to_bounding_box(np_img, 30, 635, h, w)
    return recognise_digit_image(cropped)


def recognise_digit_image(cropped, remove_first_obj=False, color_range=None):
    img = tf.keras.preprocessing.image.array_to_img(cropped)

    if color_range is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2HSV)
        img = cv2.inRange(img, color_range[0], color_range[1])
        thresh = cv2.bitwise_not(img)
    else:
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    mask = np.zeros(thresh.shape, dtype=np.float32)
    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    (cnts, _) = contours.sort_contours(cnts, method="left-to-right")
    score = ""
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 800 and area > 100:
            x, y, w, h = cv2.boundingRect(c)
            roi = 255 - thresh[y:y + h, x:x + w]
            cv2.drawContours(mask, [c], -1, (255, 255, 255), -1)
            roi = cv2.bitwise_not(roi)
            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
            score = score + read_score(roi)
    return score

def is_back_button_selected(np_img):
    back_button_color = np_img[585][700]
    # [0.07450981 0.60784316 0.00392157] highlighted
    # [0.19607843 0.6117647  0.14509805] unhighlighted
    if np.allclose(back_button_color, np.array([0.07450981, 0.60784316, 0.00392157])):
        return True
    else:
        return False



sc = capture_window()
# selected = extract_kill_game_in_progress(sc)
# selected = extract_char_count_in_progress(sc)
# selected = is_back_button_selected(sc)
selected = extract_jewel_game_in_progress(sc)
print("score: " + selected)

# cnts = grab_contours(cnts)
# digitCnts = []
# digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]
# digits = []
# # loop over the digit area candidates
# for c in cnts:
#     # compute the bounding box of the contour
#     (x, y, w, h) = cv2.boundingRect(c)
#     # if the contour is sufficiently large, it must be a digit
#     if w >= 15 and (h >= 30 and h <= 40):
#         digitCnts.append(c)
#
# # loop over each of the digits
# for c in digitCnts:
#     # extract the digit ROI
#     (x, y, w, h) = cv2.boundingRect(c)
#     roi = thresh[y:y + h, x:x + w]
#     cv2.imshow('thresh', roi)
#     cv2.waitKey(0)
#     # compute the width and height of each of the 7 segments
#     # we are going to examine
#     (roiH, roiW) = roi.shape
#     (dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
#     dHC = int(roiH * 0.05)
#     # define the set of 7 segments
#     segments = [
#         ((0, 0), (w, dH)),  # top
#         ((0, 0), (dW, h // 2)),  # top-left
#         ((w - dW, 0), (w, h // 2)),  # top-right
#         ((0, (h // 2) - dHC), (w, (h // 2) + dHC)),  # center
#         ((0, h // 2), (dW, h)),  # bottom-left
#         ((w - dW, h // 2), (w, h)),  # bottom-right
#         ((0, h - dH), (w, h))  # bottom
#     ]
#     on = [0] * len(segments)
# loop over the segments
# for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
#     # extract the segment ROI, count the total number of
#     # thresholded pixels in the segment, and then compute
#     # the area of the segment
#     segROI = roi[yA:yB, xA:xB]
#     total = cv2.countNonZero(segROI)
#     area = (xB - xA) * (yB - yA)
#     # if the total number of non-zero pixels is greater than
#     # 50% of the area, mark the segment as "on"
#     if total / float(area) > 0.5:
#         on[i] = 1
# # lookup the digit and draw it on the image
# digit = DIGITS_LOOKUP[tuple(on)]
# digits.append(digit)
# cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)
# cv2.putText(output, str(digit), (x - 10, y - 10),
#             cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

# Kill color range
# lower = np.array([155, 25, 0])
# upper = np.array([179, 255, 255])
#
# retry_stage_color = np_img[440][600]
# # [0.40784314 0.         0.06666667] highlighted
# # [0.30980393 0.10196079 0.13725491] unhighlighted
#
# shop_button_color = np_img[560][530]
# # [0.49019608 0.         0.94509804] highlighted
# # [0.42352942 0.1764706  0.6509804 ] unhighlighted
# w, h = 30, 30
# cropped = tf.image.crop_to_bounding_box(np_img, 440, 600, h, w)
# plt.imshow(cropped)
# plt.show()
# np_img = tf.keras.preprocessing.image.array_to_img(np_img)
# rgb = cv2.cvtColor(np.array(np_img), cv2.COLOR_BGR2RGB)
# b = extract_kill_game_in_progress(sc)
# print(b)


# cv2.imshow('mask', mask)
# cv2.imshow('thresh', thresh)
# cv2.waitKey(0)
# img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2HSV)
# result = img.copy()
# img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# img = cv2.inRange(img, lower, upper)
# # result = cv2.bitwise_and(result, result, mask=mask)
# # cv2.imshow('result', result)
# # img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2HSV)
# cv2.imshow('image', img)
# cv2.waitKey(0)



# cv2.imshow('image', img)
# cv2.waitKey(0)
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