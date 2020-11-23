import time

from tensorflow import keras

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from directkeys import *
from nq_screen_extractor import *


class NQEnv(py_environment.PyEnvironment):
  previous_stage = GameStage

  def __init__(self):
    self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=5, name='action')
    self._observation_spec = array_spec.BoundedArraySpec(shape=(250, 250, 3), dtype=np.float32, name='observation')
    sn = capture_window()
    sn = tf.image.resize(sn, (250, 250))
    self._state = keras.preprocessing.image.img_to_array(sn)
    self._episode_ended = False
    stage_enum = which_stage(sn)
    self.previous_stage = stage_enum

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def _reset(self): #reset this python app's state for a new game
    print("RESET")
    sn = capture_window()
    stage_enum = which_stage(sn)
    self.previous_stage = stage_enum
    sn = tf.image.resize(sn, (250, 250))
    self._state = keras.preprocessing.image.img_to_array(sn)
    self._episode_ended = False
    return ts.restart(self._state)

  def _step(self, action):
    actions = ['left_arrow', 'right_arrow', 'up_arrow', 'down_arrow', 'spacebar', 'nothing']
    if self._episode_ended:
        return self.reset()

    i = capture_window()
    stage_enum = which_stage(i)
    print("action: " + str(actions[action]) + "  which stage: " + stage_enum.name)

    img_resized = tf.image.resize(i, (250, 250))
    self._state = img_resized

    if stage_enum == GameStage.game_over:
        #add another cnn to verify quit is selected
        kill_reward = extract_kill_reward_game_over(i) / 5
        jewel_reward = extract_jewel_reward_game_over(i) / 50
        reward_game_over = kill_reward + jewel_reward
        print("game-over: k: " + str(kill_reward) + "    jewel: " + str(jewel_reward) + "   total: " + str(reward_game_over))
        time.sleep(0.30)
        self.press_spacebar()
        self.previous_stage = GameStage.game_over
        self._episode_ended = True
        return ts.termination(self._state, reward=reward_game_over)

    if action == 5 and stage_enum != GameStage.in_progress:
        return ts.transition(self._state, reward=0.0, discount=0.05)

    if action == 5:
        return ts.transition(self._state, reward=0.0, discount=0.0)

    if stage_enum == GameStage.starting_page:
        PressKey(spacebar)
        time.sleep(0.10)
        self.previous_stage = GameStage.starting_page
        return ts.transition(self._state, reward=0.0, discount=0.0)

    if stage_enum == GameStage.character_upgrade:
        PressKey(leftarrow)
        time.sleep(0.10)
        self.press_spacebar()
        self.previous_stage = GameStage.character_upgrade
        return ts.transition(self._state, reward=0.0, discount=0.05)

    if stage_enum == GameStage.interval and self.previous_stage == GameStage.in_progress:
        # extract the number of stage from image, stage * 0.5 => reward
        self.press_key(action)
        self.previous_stage = GameStage.interval
        return ts.transition(self._state, reward=0.1, discount=0.0)

    if stage_enum == GameStage.interval:
        self.press_key(action)
        self.previous_stage = GameStage.interval
        return ts.transition(self._state, reward=0.0, discount=0.0)

    if stage_enum == GameStage.interval_upgrade:
        self.press_key(action)
        self.previous_stage = GameStage.interval_upgrade
        return ts.transition(self._state, reward=0.0, discount=0.0)

    if stage_enum == GameStage.interval.interval_sorry:
        self.press_spacebar()
        self.previous_stage = GameStage.interval_sorry
        return ts.transition(self._state, reward=0.0, discount=0.005)

    if stage_enum == GameStage.in_progress and self.previous_stage == GameStage.interval:
        self.previous_stage = GameStage.in_progress
        self.press_key(action)
        return ts.transition(self._state, reward=0.005, discount=0.0)

    if stage_enum == GameStage.paused_game_while_in_progress:
        self.press_spacebar()
        return ts.transition(self._state, reward=0.0, discount=0.05)

    if stage_enum == GameStage.in_progress:
        if self.previous_stage == GameStage.in_progress and action != 4:
            self.press_key(action)
            return ts.transition(self._state, reward=0.005, discount=0.0)

        self.previous_stage = GameStage.in_progress
        if action == 4:
            return ts.transition(self._state, reward=0.0, discount=0.05)
        self.press_key(action)
        return ts.transition(self._state, reward=0.0, discount=0.0)

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

  def press_spacebar(self):
      PressKey(spacebar)
      time.sleep(0.05)

  def press_key(self, action):
      keys_to_press = [[leftarrow], [rightarrow], [uparrow], [downarrow], [spacebar]]
      if action != 5:
          for key in keys_to_press[action]:
              PressKey(key)
              time.sleep(0.05)
