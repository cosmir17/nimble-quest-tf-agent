from pynput.keyboard import Key, Controller
from key_pressor.AbstractKeyPressor import AbstractKeyPressor
import time


class WindowKeyPressor(AbstractKeyPressor):

    @property
    def keyboard(self):
        return Controller()

    @property
    def spacebar(self):
        return Key.space

    @property
    def leftarrow(self):
        return Key.left

    @property
    def rightarrow(self):
        return Key.right

    @property
    def uparrow(self):
        return Key.up

    @property
    def downarrow(self):
        return Key.down

    @property
    def enter(self):
        return Key.enter

    def press_key(self, key):
        self.keyboard.press(key)
        time.sleep(0.005)
        self.keyboard.release(key)
        time.sleep(0.005)
