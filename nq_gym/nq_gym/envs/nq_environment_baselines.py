import time
import random
import gym
from gym import spaces

from key_pressor.key_pressor_creator import *
from nq_screen_extractor import *
from tensorflow import keras

screenshot_upper_bound = 90000

jewel_reward_point = 0.00000
kill_reward_point = 0.0000
reward_for_being_alive = 0.0


class NQEnv(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self):
        super(NQEnv, self).__init__()
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=255, shape=(130, 130, 1), dtype=np.uint8)
        screenshot, stage_enum = self.take_screenshot_save_to_selfstate()
        self._episode_ended = False
        self._stage = stage_enum
        self._raw_screenshot = screenshot
        self._infinite_loop_safe_guard = 0
        self._first_game_stage_not_finished = True
        self._game_over_penalty_is_given = False
        self.key_pressor = create_key_pressor()

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

        actions = ['left_arrow', 'right_arrow', 'up_arrow', 'down_arrow', 'nothing']
        try:
            str(actions[action])
            print("action: " + str(actions[action]) + "  which stage: " + str(self._stage))
        except:
            print("only integer scalar array error: " + str(action))
            self.take_screenshot_save_to_selfstate()
            return self._state, 0.0, False, {}

        if (self._stage == GameStage.game_over or self._stage == GameStage.died) and is_back_button_selected(self._raw_screenshot):
            time.sleep(0.5)
            self.press_spacebar()
            time.sleep(0.5)
            screenshot, stage_enum = self.take_screenshot_save_to_selfstate()
            self._episode_ended = True # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            if not self._game_over_penalty_is_given:
                tf.keras.preprocessing.image.save_img("backbutton_without_panelty_c_" + str(random.randint(0, screenshot_upper_bound)) + ".png", self._raw_screenshot, file_format='png')
                tf.keras.preprocessing.image.save_img("backbutton_without_panelty_next_" + str(random.randint(0, screenshot_upper_bound)) + ".png", screenshot, file_format='png')
                self._stage = stage_enum
                self._raw_screenshot = screenshot
                print("game_over_penalty_ was not _given, applying penalty") #it's game start page
                return self._state, -2.0, True, {}
            else:
                self._stage = stage_enum
                self._raw_screenshot = screenshot
                return self._state, 0.0, True, {}

        if self._stage == GameStage.game_over or self._stage == GameStage.died:
            time.sleep(0.5)
            self.press_key(2)
            time.sleep(0.5)
            screenshot, stage_enum = self.take_screenshot_save_to_selfstate()
            if stage_enum == GameStage.in_progress:
                tf.keras.preprocessing.image.save_img("gameover_but_inprogress_c" + str(random.randint(0, screenshot_upper_bound)) + ".png", self._raw_screenshot, file_format='png')
                tf.keras.preprocessing.image.save_img("gameover_but_inprogress_next" + str(random.randint(0, screenshot_upper_bound)) + ".png", screenshot, file_format='png')
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

        if self._stage == GameStage.interval:
            time.sleep(0.1)
            self.press_key(3)
            time.sleep(0.1)
            self.press_spacebar()
            time.sleep(0.3)
            screenshot, stage_enum = self.take_screenshot_save_to_selfstate()
            self._stage = stage_enum
            self._raw_screenshot = screenshot
            return self._state, 0.0, False, {}

        if self._stage == GameStage.interval_upgrade:
            time.sleep(0.1)
            self.press_key(0)
            time.sleep(0.1)
            self.press_spacebar()
            time.sleep(0.1)
            self._infinite_loop_safe_guard = self._infinite_loop_safe_guard + 1
            screenshot, stage_enum = self.take_screenshot_save_to_selfstate()
            self._stage = stage_enum
            self._raw_screenshot = screenshot
            return self._state, 0.0, False, {}

        if self._stage == GameStage.interval_sorry:
            self.press_spacebar()
            time.sleep(0.1)
            screenshot, stage_enum = self.take_screenshot_save_to_selfstate()
            self._stage = stage_enum
            self._raw_screenshot = screenshot
            return self._state, 0.0, False, {}

        if self._stage == GameStage.game_over_sorry:
            self.press_spacebar()
            time.sleep(0.1)
            screenshot, stage_enum = self.take_screenshot_save_to_selfstate()
            self._stage = stage_enum
            self._raw_screenshot = screenshot
            return self._state, 0.0, False, {}

        if self._stage == GameStage.store_page:
            self.press_key(3)  # select back button
            time.sleep(0.1)
            self.press_spacebar()
            time.sleep(0.1)
            screenshot, stage_enum = self.take_screenshot_save_to_selfstate()
            self._stage = stage_enum
            self._raw_screenshot = screenshot
            self._episode_ended = True # # # # # # # # because it can miss clicking the backbutton
            if self._game_over_penalty_is_given:
                return self._state, 0.0, True, {}
            else:
                return self._state, -2.0, True, {}

        if self._stage == GameStage.main_page:
            self.press_spacebar()
            time.sleep(0.1)
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
                return self._state, -2.0, False, {}
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
            return self._state, 0.0, False, {}

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
            if next_stage_enum == GameStage.game_over:
                print("game over without death scene")
                tf.keras.preprocessing.image.save_img("c_without_death_scene_" + str(random.randint(0, screenshot_upper_bound)) + ".png", self._raw_screenshot, file_format='png')
                tf.keras.preprocessing.image.save_img("n_without_death_scene_" + str(random.randint(0, screenshot_upper_bound)) + ".png", next_screenshot, file_format='png')
            self._stage = next_stage_enum
            self._raw_screenshot = next_screenshot
            # print("before game is ended, stage: " + str(next_stage_enum))
            self._game_over_penalty_is_given = True
            return self._state, -2.0, False, {}

        if self._stage == GameStage.in_progress and next_stage_enum == GameStage.interval:
            print("**finished stage")
            self._stage = next_stage_enum
            self._raw_screenshot = next_screenshot
            self._first_game_stage_not_finished = False
            return self._state, 1.0, False, {}

        if self._stage == GameStage.in_progress:
            print("unexpected route - in progress => " + next_stage_enum.name)
            tf.keras.preprocessing.image.save_img(
                "unexpected_progress_c" + str(random.randint(0, screenshot_upper_bound)) + ".png",
                self._raw_screenshot, file_format='png')
            tf.keras.preprocessing.image.save_img(
                "unexpected_progress_n" + str(random.randint(0, screenshot_upper_bound)) + ".png", next_screenshot,
                file_format='png')
            self._stage = next_stage_enum
            self._raw_screenshot = next_screenshot
            self._game_over_penalty_is_given = True
            return self._state, -2.0, False, {}

        self._game_over_penalty_is_given = True
        print("unexpected route ****** which stage in next turn: " + next_stage_enum.name)
        tf.keras.preprocessing.image.save_img(
            "unexpected_final_progress_c" + str(random.randint(0, screenshot_upper_bound)) + ".png",
            self._raw_screenshot, file_format='png')
        tf.keras.preprocessing.image.save_img(
            "unexpected_final_progress_n" + str(random.randint(0, screenshot_upper_bound)) + ".png", next_screenshot,
            file_format='png')
        self._stage = next_stage_enum
        self._raw_screenshot = next_screenshot
        return self._state, -2.0, False, {}

    def take_screenshot_save_to_selfstate(self):
        i = capture_window()
        stage_enum = which_stage(i)
        img_resized = tf.image.resize(i, (130, 130))
        img_resized = tf.image.rgb_to_grayscale(img_resized)
        img_resized = keras.preprocessing.image.img_to_array(img_resized)
        img_resized_255 = img_resized * 255
        img_resized_255 = img_resized_255.astype(np.uint8)
        self._state = img_resized_255
        return i, stage_enum

    def calculate_reward_game_in_progress(self, screenshot, stage_enum, screenshot_after_action, next_stage_enum):
        return reward_for_being_alive

    def press_spacebar(self):
        self.key_pressor.press_key(self.key_pressor.spacebar)
        time.sleep(0.1)

    def press_key(self, action):
        keys_to_press = [[self.key_pressor.leftarrow], [self.key_pressor.rightarrow], [self.key_pressor.uparrow], [self.key_pressor.downarrow]]
        if action != 4:
            for key in keys_to_press[action]:
                self.key_pressor.press_key(key)
        else:
            time.sleep(0.01)
