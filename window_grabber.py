from PIL import ImageGrab
import win32gui # install pywin32 if you are running on windows
import sys


def place_resize_nq_window():
    hwnd = win32gui.FindWindow(None, 'Nimble Quest')
    if hwnd == 0:
        sys.exit('no nimble quest window is present')
    x0, y0, x1, y1 = win32gui.GetWindowRect(hwnd)
    if x0 != 0 or y0 != 0 or x1 != 1261 or y1 != 702:
        print("Nimble Quest Window is not located correctly or the size is wrong, adjusting automatically")
        win32gui.MoveWindow(hwnd, 0, 0, 1275, 725, True)

    win32gui.SetForegroundWindow(hwnd)
    bbox = win32gui.GetWindowRect(hwnd)
    return bbox
