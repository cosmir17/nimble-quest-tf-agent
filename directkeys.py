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


# def ReleaseKey(key):
#     extra = ctypes.c_ulong(0)
#     ii_ = Input_I()
#     ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra))
#     x = Input(ctypes.c_ulong(1), ii_)
#     ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


# if __name__ == '__main__':
#     PressKey(0x11)
#     time.sleep(1)
#     ReleaseKey(0x11)
#     time.sleep(1)
