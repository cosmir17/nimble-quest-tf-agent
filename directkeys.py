import os

from pynput.keyboard import Key, Controller
import platform
import time


spacebar = (49, Key.space)
leftarrow = (123, Key.left)
rightarrow = (124, Key.right)
uparrow = (126, Key.up)
downarrow = (125, Key.down)
enter = (36, Key.enter)

os_name = platform.system()

if os_name == "Linux":
    keyboard = Controller()

def PressKey(key):
    if os_name == "Darwin":
        cmd = "osascript -e 'tell application \"System Events\" to key code \"" + str(key[0]) + "\"'"
        os.system(cmd)
        time.sleep(0.05)
    elif os_name == "Linux":
        keyboard.press(key[1])
        time.sleep(0.025)
        keyboard.release(key[1])
        time.sleep(0.025)