from pynput import keyboard
from threading import Thread
from time import sleep
import sys
import os
import psutil

def on_press(key):
    if key == keyboard.Key.esc:
        print('ESC pressed')
        os._exit(0)
        # quit()
        # raise SystemExit
        # os._exit(0)
        current_system_pid = os.getpid()
        this_system = psutil.Process(current_system_pid)
        this_system.terminate()


def on_release(key):
    if key == keyboard.Key.esc:
        print('ESC released')
        os._exit(0)
        # quit()
        # raise SystemExit
        # os._exit(0)
        current_system_pid = os.getpid()
        this_system = psutil.Process(current_system_pid)
        this_system.terminate()


def run_terminator_listener():
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

# def threaded_function(arg):
#     for i in range(arg):
#         print("running")
#         sleep(1)
#
# thread = Thread(target=threaded_function, args=(10,))
# thread.start()
# thread.join()
# print("thread finished...exiting")