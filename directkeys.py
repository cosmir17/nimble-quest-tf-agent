import os

spacebar = 49
leftarrow = 123
rightarrow = 124
uparrow = 126
downarrow = 125
enter = 36


def PressKey(key):
    cmd = "osascript -e 'tell application \"System Events\" to key code \"" + str(key) + "\"'"
    os.system(cmd)
