import time

from tensorflow import keras

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from directkeys import *
from nq_screen_extractor import *


class NQEnv(py_environment.PyEnvironment):

  def __init__(self):
    self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=5, name='action')
    self._observation_spec = array_spec.BoundedArraySpec(shape=(70, 70, 1), dtype=np.float32, name='observation')
    sn = capture_window()
    resized = tf.image.resize(sn, (70, 70))
    resized = tf.image.rgb_to_grayscale(resized)
    self._state = keras.preprocessing.image.img_to_array(resized)
    self._episode_ended = False
    stage_enum = which_stage(sn)
    self.previous_stage = stage_enum
    self.previous_kill_count = None
    self.previous_jewel_count = None
    self.infinite_loop_safe_guard = 0

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def _reset(self): #reset this python app's state for a new game
    print("RESET")
    sn = capture_window()
    stage_enum = which_stage(sn)
    self.previous_stage = stage_enum
    resized = tf.image.resize(sn, (70, 70))
    resized = tf.image.rgb_to_grayscale(resized)
    self._state = keras.preprocessing.image.img_to_array(resized)
    self._episode_ended = False
    self.previous_kill_count = None
    self.previous_jewel_count = None
    self.infinite_loop_safe_guard = 0
    return ts.restart(self._state)

  def _step(self, action):
    actions = ['left_arrow', 'right_arrow', 'up_arrow', 'down_arrow', 'spacebar', 'nothing']
    if self._episode_ended:
        return self.reset()

    i = capture_window()
    stage_enum = which_stage(i)
    print("action: " + str(actions[action]) + "  which stage: " + stage_enum.name)

    img_resized = tf.image.resize(i, (70, 70))
    img_resized = tf.image.rgb_to_grayscale(img_resized)
    img_resized = keras.preprocessing.image.img_to_array(img_resized)
    self._state = img_resized

    if stage_enum == GameStage.game_over and not is_back_button_selected(i):
        self.previous_stage = GameStage.game_over
        self.infinite_loop_safe_guard = self.infinite_loop_safe_guard + 1
        penalty = self.infinite_loop_safe_guard * 0.05
        if self.infinite_loop_safe_guard > 60:
            self.press_key(2)
            return ts.transition(self._state, reward=0.0, discount=penalty)
        if action == 4 or action == 5:
            return ts.transition(self._state, reward=0.0, discount=penalty)
        self.press_key(action)
        return ts.transition(self._state, reward=0.0, discount=penalty)

    if stage_enum == GameStage.game_over and is_back_button_selected(i):
        kill_reward = extract_kill_reward_game_over(i)
        jewel_reward = extract_jewel_reward_game_over(i)
        if kill_reward is None:
            kill_reward = 0
        else:
            kill_reward = kill_reward * 0.05
        if jewel_reward is None:
            jewel_reward = 0
        else:
            jewel_reward = jewel_reward * 0.005
        # reward_game_over = kill_reward + jewel_reward
        # print("game-over: k: " + str(kill_reward) + "    jewel: " + str(jewel_reward) + "   total: " + str(reward_game_over))
        time.sleep(0.10)
        self.press_spacebar()
        self.previous_stage = GameStage.game_over
        self._episode_ended = True
        return ts.termination(self._state, reward=0.0)

    if stage_enum == GameStage.starting_page:
        self.press_spacebar()
        self.previous_stage = GameStage.starting_page
        return ts.transition(self._state, reward=0.0, discount=0.0)

    if stage_enum == GameStage.character_upgrade:
        PressKey(leftarrow)
        time.sleep(0.10)
        self.press_spacebar()
        self.previous_stage = GameStage.character_upgrade
        return ts.transition(self._state, reward=0.0, discount=0.05)

    if stage_enum == GameStage.interval and self.previous_stage == GameStage.in_progress:
        self.infinite_loop_safe_guard = self.infinite_loop_safe_guard + 1
        self.press_key(action)
        self.previous_stage = GameStage.interval
        return ts.transition(self._state, reward=0.1, discount=0.0)

    if stage_enum == GameStage.interval:
        self.previous_stage = GameStage.interval
        if self.infinite_loop_safe_guard == 0:
            self.press_spacebar()
        else:
            self.infinite_loop_safe_guard = self.infinite_loop_safe_guard + 1
            self.press_key(action)
            time.sleep(0.1)
        if self.infinite_loop_safe_guard > 60:
            print("looping in Interval stage for more than 60 times")
            self._episode_ended = True
            return ts.termination(self._state, reward=0.0)
        return ts.transition(self._state, reward=0.0, discount=self.infinite_loop_safe_guard * 0.0005)

    if stage_enum == GameStage.interval_upgrade:
        self.press_key(action)
        self.previous_stage = GameStage.interval_upgrade
        self.infinite_loop_safe_guard = self.infinite_loop_safe_guard + 1
        if self.infinite_loop_safe_guard > 60:
            print("looping in Interval stage for more than 60 times")
            self._episode_ended = True
            return ts.termination(self._state, reward=0.0)
        return ts.transition(self._state, reward=0.0, discount=self.infinite_loop_safe_guard * 0.0005)

    if stage_enum == GameStage.interval.interval_sorry:
        self.press_spacebar()
        self.previous_stage = GameStage.interval_sorry
        return ts.transition(self._state, reward=0.0, discount=0.005)

    if stage_enum == GameStage.in_progress and self.previous_stage == GameStage.interval:
        if self.infinite_loop_safe_guard != 0:
            self.infinite_loop_safe_guard = 0
        reward = self.calculate_reward_game_in_progress(i)
        self.previous_stage = GameStage.in_progress
        self.press_key(action)
        return ts.transition(self._state, reward=reward, discount=0.0)

    if stage_enum == GameStage.paused_game_while_in_progress:
        self.previous_stage = GameStage.paused_game_while_in_progress
        self.press_spacebar()
        return ts.transition(self._state, reward=0.0, discount=0.05)

    if stage_enum == GameStage.in_progress:
        self.previous_stage = GameStage.in_progress
        if action == 4:
            time.sleep(0.05)
            return ts.transition(self._state, reward=0.0, discount=0.05)
        else:
            reward = self.calculate_reward_game_in_progress(i)
            self.press_key(action)
            return ts.transition(self._state, reward=reward, discount=0.0)

    if stage_enum == GameStage.game_over_sorry:
        self.previous_stage = GameStage.game_over_sorry
        self.press_spacebar()
        return ts.transition(self._state, reward=0.0, discount=0.05)

    if stage_enum == GameStage.store_page:
        self.previous_stage = GameStage.store_page
        self.press_key(3) #select back button
        self.press_spacebar()
        self._episode_ended = True
        return ts.transition(self._state, reward=0.0, discount=0.05)

    if stage_enum == GameStage.main_page:
        self.previous_stage = GameStage.main_page
        self.press_spacebar()
        return ts.transition(self._state, reward=0.0, discount=0.0)

    print("unexpected route ******  action: " + str(actions[action]) + "  which stage: " + stage_enum.name)
    return ts.transition(self._state, reward=0.0, discount=0.0)

  def calculate_reward_game_in_progress(self, i):
      kill_no = extract_kill_game_in_progress(i)
      jewel_no = extract_jewel_game_in_progress(i)
      # print("prev_kill_no: " + str(self.previous_kill_count) + "  kill no: " + str(kill_no))
      # print("prev_jewel_count: " + str(self.previous_jewel_count) + " jewel no: " + str(jewel_no))
      ########################
      if self.previous_kill_count is None:
          p_kill_count = 0
      else:
          p_kill_count = self.previous_kill_count

      if self.previous_jewel_count is None:
          p_jewel_count = 0
      else:
          p_jewel_count = self.previous_jewel_count
      #########################
      if kill_no is None and jewel_no is None:
        return 0.005
      elif kill_no is not None and jewel_no is not None:
          kill_diff = kill_no - p_kill_count
          jewel_diff = jewel_no - p_jewel_count
          if kill_diff > 10 and jewel_diff > 70:
              return 0.005
          elif kill_diff > 10:
              self.previous_jewel_count = jewel_no
              return (jewel_diff * 0.005) + 0.005
          elif jewel_diff > 70:
              self.previous_kill_count = kill_no
              return (kill_diff * 0.05) + 0.005
          else:
              self.previous_kill_count = kill_no
              self.previous_jewel_count = jewel_no
              return (kill_diff * 0.05) + (jewel_diff * 0.005) + 0.005
      elif jewel_no is None:
          self.previous_kill_count = kill_no
          kill_diff = kill_no - p_kill_count
          return (kill_diff * 0.05) + 0.005
      elif kill_no is None:
          self.previous_jewel_count = jewel_no
          jewel_diff = jewel_no - p_jewel_count
          return (jewel_diff * 0.005) + 0.005


  def press_spacebar(self):
      PressKey(spacebar)
      time.sleep(0.1)

  def press_key(self, action):
      keys_to_press = [[leftarrow], [rightarrow], [uparrow], [downarrow], [spacebar]]
      if action != 5:
          for key in keys_to_press[action]:
              PressKey(key)
      time.sleep(0.05)
