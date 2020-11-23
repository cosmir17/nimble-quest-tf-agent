import os

import psutil
from pynput import keyboard


def on_press(key):
    if key == keyboard.Key.esc:
        print('ESC pressed')
        os._exit(0)
        current_system_pid = os.getpid()
        this_system = psutil.Process(current_system_pid)
        this_system.terminate()


def on_release(key):
    if key == keyboard.Key.esc:
        print('ESC released')
        os._exit(0)
        current_system_pid = os.getpid()
        this_system = psutil.Process(current_system_pid)
        this_system.terminate()


def run_terminator_listener():
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
