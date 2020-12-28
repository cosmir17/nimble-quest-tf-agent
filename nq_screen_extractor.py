# import cv2
import numpy as np
# import pytesseract as pt
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from mss import mss
from tensorflow.keras.models import load_model
# from imutils import grab_contours, contours
from screen_classifier.stage_classifier.game_stage import GameStage

stage_weight_path = "screen_classifier/stage_classifier/nq_screen_weight.h5"
stage_model = load_model(stage_weight_path)


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
    img_resized = tf.image.resize(img, (100, 100)) #stage
    # prediction = stage_model.predict(np.array([img_resized]))
    prediction = stage_model.predict_on_batch(np.array([img_resized]))
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


def is_back_button_selected(np_img):
    back_button_color = np_img[585][700]
    # [0.07450981 0.60784316 0.00392157] highlighted
    # [0.19607843 0.6117647  0.14509805] unhighlighted
    if np.allclose(back_button_color, np.array([0.07450981, 0.60784316, 0.00392157]),
                   rtol=1.e-1, atol=1.e-1):
        return True
    else:
        return False


captured = capture_window()
print("backbutton selected:" + str(is_back_button_selected(captured)))
plt.imshow(captured)
plt.show()

# prediction = tf.image.decode_png(tf.io.read_file("wrong_kill_count_60.png"), channels=3)
# prediction = tf.keras.preprocessing.image.img_to_array(prediction)
# # # selected = is_back_button_selected(prediction)
# selected = extract_kill_game_in_progress(prediction)
# print("score: " + str(selected))
