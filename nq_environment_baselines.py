import time

import gym
from gym import spaces

from directkeys import *
from nq_screen_extractor import *
from tensorflow import keras


class NQEnv(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self):
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=0, high=255, shape=(70, 70, 1), dtype=np.uint8)
        screenshot, stage_enum = self.take_screenshot_save_to_selfstate()
        self._episode_ended = False
        self._stage = stage_enum
        self._raw_screenshot = screenshot
        self._infinite_loop_safe_guard = 0
        self._first_game_stage_not_finished = True
        self._game_over_penalty_is_given = False

    def reset(self): #reset this python app's state for a new game
        print("RESET")
        screenshot, stage_enum = self.take_screenshot_save_to_selfstate()
        self._stage = stage_enum
        self._raw_screenshot = screenshot
        self._episode_ended = False
        self._infinite_loop_safe_guard = 0
        self._first_game_stage_not_finished = True
        self._game_over_penalty_is_given = False
        return self._state

    def step(self, action):
        if self._episode_ended:
           return self.reset()

        actions = ['left_arrow', 'right_arrow', 'up_arrow', 'down_arrow', 'spacebar', 'nothing']
        try:
            print("action: " + str(actions[action]) + "  which stage: " + str(self._stage))
        except:
            print("only integer scalar array error: " + str(action))
            self.take_screenshot_save_to_selfstate()
            return self._state, 0.0, False, {}

        if (self._stage == GameStage.game_over or self._stage == GameStage.died) and is_back_button_selected(self._raw_screenshot):
            self.press_spacebar()
            time.sleep(0.3)
            screenshot, stage_enum = self.take_screenshot_save_to_selfstate()
            self._episode_ended = True
            if not self._game_over_penalty_is_given:
                tf.keras.preprocessing.image.save_img("current_2.png", self._raw_screenshot, file_format='png')
                tf.keras.preprocessing.image.save_img("next_stage_2.png", screenshot, file_format='png')
                self._stage = stage_enum
                self._raw_screenshot = screenshot
                print("game_over_penalty_ was not _given, applying penalty")
                return self._state, -2.0, True, {}
            else:
                self._stage = stage_enum
                self._raw_screenshot = screenshot
                return self._state, 0.0, True, {}

        if self._stage == GameStage.game_over or self._stage == GameStage.died:
            time.sleep(0.1)
            self.press_key(2)
            screenshot, stage_enum = self.take_screenshot_save_to_selfstate()
            if stage_enum == GameStage.in_progress:
                tf.keras.preprocessing.image.save_img("current.png", self._raw_screenshot, file_format='png')
                tf.keras.preprocessing.image.save_img("next_stage.png", screenshot, file_format='png')
                print("*** CNN game over recognition error ***, stage: " + str(self._stage) + " next stage: in_progress")
            self._stage = stage_enum
            self._raw_screenshot = screenshot
            return self._state, 0.0, False, {}

        if self._stage == GameStage.starting_page:
            time.sleep(0.1)
            self.press_spacebar()
            time.sleep(0.1)
            screenshot, stage_enum = self.take_screenshot_save_to_selfstate()
            self._stage = stage_enum
            self._raw_screenshot = screenshot
            return self._state, 0.0, False, {}

        if self._stage == GameStage.character_upgrade:
            time.sleep(0.1)
            self.press_key(0)
            time.sleep(0.1)
            self.press_spacebar()
            screenshot, stage_enum = self.take_screenshot_save_to_selfstate()
            self._stage = stage_enum
            self._raw_screenshot = screenshot
            return self._state, 0.0, False, {}

        if self._stage == GameStage.interval and self._first_game_stage_not_finished:
            self.press_spacebar()
            time.sleep(0.2)
            screenshot, stage_enum = self.take_screenshot_save_to_selfstate()
            self._stage = stage_enum
            self._raw_screenshot = screenshot
            return self._state, 0.0, False, {}

        if self._stage == GameStage.interval: #after first game stage is done
            self._infinite_loop_safe_guard = self._infinite_loop_safe_guard + 1
            self.press_key(action)
            time.sleep(0.2)
            screenshot, stage_enum = self.take_screenshot_save_to_selfstate()
            self._stage = stage_enum
            self._raw_screenshot = screenshot
            if stage_enum == GameStage.in_progress:
                self._infinite_loop_safe_guard = 0
                return self._state, 0.005, False, {}
            elif self._infinite_loop_safe_guard > 60:
                print("looping in Interval stage for more than 60 times")
                self._episode_ended = True
                return self._state, -0.01, True, {}
            else:
               return self._state, -(self._infinite_loop_safe_guard * 0.0008), False, {}

        if self._stage == GameStage.interval_upgrade:
            self.press_key(action)
            self._infinite_loop_safe_guard = self._infinite_loop_safe_guard + 1
            screenshot, stage_enum = self.take_screenshot_save_to_selfstate()
            self._stage = stage_enum
            self._raw_screenshot = screenshot
            if self._infinite_loop_safe_guard > 60:
                print("looping in Interval stage for more than 60 times")
                self._episode_ended = True
                return self._state, -0.01, True, {}
            if stage_enum == GameStage.interval_upgrade:
                return self._state, -(self._infinite_loop_safe_guard * 0.0008), False, {}
            if stage_enum == GameStage.interval:
                return self._state, 0.0, False, {}
            if stage_enum == GameStage.interval_sorry:
                return self._state, -0.001, False, {}
            else:
                print("mostly likely, game over, a penalty is given, inside "
                      "interval_upgrade: next stage: " + str(stage_enum))
                return self._state, -2.0, False, {}

        if self._stage == GameStage.interval_sorry:
            self.press_spacebar()
            screenshot, stage_enum = self.take_screenshot_save_to_selfstate()
            self._stage = stage_enum
            self._raw_screenshot = screenshot
            return self._state, 0.0, False, {}

        if self._stage == GameStage.game_over_sorry:
            self.press_spacebar()
            screenshot, stage_enum = self.take_screenshot_save_to_selfstate()
            self._stage = stage_enum
            self._raw_screenshot = screenshot
            return self._state, 0.0, False, {}

        if self._stage == GameStage.store_page:
            self.press_key(3)  # select back button
            self.press_spacebar()
            screenshot, stage_enum = self.take_screenshot_save_to_selfstate()
            self._stage = stage_enum
            self._raw_screenshot = screenshot
            self._episode_ended = True
            return self._state, 0.0, False, {}

        if self._stage == GameStage.main_page:
            self.press_spacebar()
            screenshot, stage_enum = self.take_screenshot_save_to_selfstate()
            self._stage = stage_enum
            self._raw_screenshot = screenshot
            return self._state, 0.0, False, {}

        if self._stage == GameStage.paused_game_while_in_progress:
            self.press_spacebar()
            time.sleep(0.15)
            screenshot, stage_enum = self.take_screenshot_save_to_selfstate()
            if stage_enum == GameStage.died or stage_enum == GameStage.game_over:
                self._game_over_penalty_is_given = True
                # print("game over penalty is given, next stage: " + str(stage_enum))
                self._stage = stage_enum
                self._raw_screenshot = screenshot
                return self._state, -1.0, False, {}
            self._stage = stage_enum
            self._raw_screenshot = screenshot
            return self._state, 0.0, False, {}

    #######################################
        self.press_key(action)

        next_screenshot, next_stage_enum = self.take_screenshot_save_to_selfstate()
        # print(" next stage: " + str(next_stage_enum))

        if self._stage == GameStage.in_progress and next_stage_enum == GameStage.paused_game_while_in_progress:
            self._stage = next_stage_enum
            self._raw_screenshot = next_screenshot
            return self._state, -5.0, False, {}

        if self._stage == GameStage.in_progress and next_stage_enum == GameStage.in_progress:
            if self._infinite_loop_safe_guard != 0:
                self._infinite_loop_safe_guard = 0
            reward = self.calculate_reward_game_in_progress(self._raw_screenshot, self._stage, next_screenshot, next_stage_enum)
            self._stage = next_stage_enum
            self._raw_screenshot = next_screenshot
            return self._state, reward, False, {}

        if self._stage == GameStage.in_progress \
                and (next_stage_enum == GameStage.game_over
                         or next_stage_enum == GameStage.interval_upgrade
                         or next_stage_enum == GameStage.died
                         or next_stage_enum == GameStage.starting_page):
            self._stage = next_stage_enum
            self._raw_screenshot = next_screenshot
            # print("game over penalty is given, next stage: " + str(next_stage_enum))
            self._game_over_penalty_is_given = True
            return self._state, -2.0, False, {}

        if self._stage == GameStage.in_progress and next_stage_enum == GameStage.interval:
            print("**finished stage")
            self._stage = next_stage_enum
            self._raw_screenshot = next_screenshot
            return self._state, 0.2, False, {}

        if self._stage == GameStage.in_progress:
            self._stage = next_stage_enum
            self._raw_screenshot = next_screenshot
            print("unexpected route - in progress => " + next_stage_enum.name)
            self._game_over_penalty_is_given = True
            return self._state, -2.0, False, {}

        self._stage = next_stage_enum
        self._raw_screenshot = next_screenshot
        print("unexpected route ****** which stage in next turn: " + next_stage_enum.name)
        return self._state, 0.0, False, {}

    def take_screenshot_save_to_selfstate(self):
        i = capture_window()
        stage_enum = which_stage(i)
        img_resized = tf.image.resize(i, (70, 70))
        img_resized = tf.image.rgb_to_grayscale(img_resized)
        img_resized = keras.preprocessing.image.img_to_array(img_resized)
        img_resized_255 = img_resized * 255
        img_resized_255 = img_resized_255.astype(np.uint8)
        self._state = img_resized_255
        return i, stage_enum

    def calculate_reward_game_in_progress(self, screenshot, stage_enum, screenshot_after_action, next_stage_enum):
        kill_no = extract_kill_game_in_progress(screenshot)
        jewel_no = extract_jewel_game_in_progress(screenshot)
        next_kill_no = extract_kill_game_in_progress(screenshot_after_action)
        next_jewel_no = extract_jewel_game_in_progress(screenshot_after_action)

        if kill_no is None and jewel_no is None and next_kill_no is None and next_jewel_no is None:
            return 0.05
        elif all(v is not None for v in [kill_no, jewel_no, next_kill_no, next_jewel_no]):
            kill_diff = next_kill_no - kill_no
            if kill_diff < 0 or kill_diff > 10:
                   kill_diff = 0
            jewel_diff = next_jewel_no - jewel_no
            if jewel_diff < 0 or jewel_diff > 70:
                   jewel_diff = 0
            reward = (kill_diff * 0.01) + (jewel_diff * 0.001) + 0.05
            if kill_no > 10 :
                   tf.keras.preprocessing.image.save_img("wrong_kill_count_" + str(kill_no) + ".png", screenshot, file_format='png')
                   print("reward: " + str(reward) + " killno: " + str(kill_no) + " jewel_no: " + str(jewel_no) +
                          " next Kill_no: " + str(next_kill_no) + " next jewel_no: " + str(next_jewel_no))
            return reward
        elif (kill_no is None or next_kill_no is None) and jewel_no is not None and next_jewel_no is not None:
            jewel_diff = next_jewel_no - jewel_no
            if jewel_diff < 0 or jewel_diff > 70:
                  jewel_diff = 0
            return (jewel_diff * 0.001) + 0.05
        elif kill_no is not None and next_kill_no is not None and (jewel_no is None or next_jewel_no is None):
            kill_diff = next_kill_no - kill_no
            if kill_diff < 0 or kill_diff > 10:
                   kill_diff = 0
            return (kill_diff * 0.01) + 0.05
        else:
            print("cnn recognition error: killno: " + str(kill_no) + " jewel_no: " + str(jewel_no) +
                  " next Kill_no" + str(next_kill_no) + " next jewel_no: " + str(next_jewel_no))
            return 0.05

    def press_spacebar(self):
        PressKey(spacebar)
        time.sleep(0.1)

    def press_key(self, action):
        keys_to_press = [[leftarrow], [rightarrow], [uparrow], [downarrow], [spacebar]]
        if action != 5:
             for key in keys_to_press[action]:
                  PressKey(key)
                  time.sleep(0.05)
        else:
             time.sleep(0.07)
