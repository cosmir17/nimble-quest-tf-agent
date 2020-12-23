from key_pressor.AbstractKeyPressor import AbstractKeyPressor
import time
import os


class MacKeyPressor(AbstractKeyPressor):

    @property
    def spacebar(self):
        return 49

    @property
    def leftarrow(self):
        return 123

    @property
    def rightarrow(self):
        return 124

    @property
    def uparrow(self):
        return 126

    @property
    def downarrow(self):
        return 125

    @property
    def enter(self):
        return 36

    def press_key(self, key):
        cmd = "osascript -e 'tell application \"System Events\" to key code \"" + str(key) + "\"'"
        os.system(cmd)
        time.sleep(0.01)
