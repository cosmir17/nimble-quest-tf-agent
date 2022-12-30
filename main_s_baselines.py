from stable_baselines3 import PPO
# from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3.a2c.policies import CnnPolicy
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.env_checker import check_env
from nq_environment_baselines import *
from power_switch import *

run_terminator_listener()

# import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
#   except RuntimeError as e:
#     print(e)

env = DummyVecEnv([lambda: NQEnv()])
# Train the agent
model = PPO(ActorCriticCnnPolicy, env, verbose=2).learn(total_timesteps=50000, log_interval=50000)
model.save("nimble_quest_stable_baselines_ppo_20000_1")


# for i in range(30):
#     if i == 0:
#         model = PPO.load("nimble_quest_stable_baselines_ppo_20000", tensorboard_log="my_tf_board")
#     else:
#         model = PPO.load("nimble_quest_stable_baselines_ppo_20000_" + str(i), tensorboard_log="my_tf_board")
#     model.set_env(env)
#     model.learn(total_timesteps=20000, log_interval=1, reset_num_timesteps=False)
#     model.save("nimble_quest_stable_baselines_ppo_20000_" + str(i + 1))
#     mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

model = PPO.load("nimble_quest_stable_baselines_ppo_20000_1", tensorboard_log="my_tf_board")
model.set_env(env)
model.learn(total_timesteps=30000, log_interval=30000, reset_num_timesteps=False)
model.save("nimble_quest_stable_baselines_ppo_20000_2")

model = PPO.load("nimble_quest_stable_baselines_ppo_20000_2", tensorboard_log="my_tf_board")
model.set_env(env)
model.learn(total_timesteps=30000, log_interval=30000, reset_num_timesteps=False)
model.save("nimble_quest_stable_baselines_ppo_20000_3")

model = PPO.load("nimble_quest_stable_baselines_ppo_20000_3", tensorboard_log="my_tf_board")
model.set_env(env)
model.learn(total_timesteps=30000, log_interval=30000, reset_num_timesteps=False)
model.save("nimble_quest_stable_baselines_ppo_20000_4")

model = PPO.load("nimble_quest_stable_baselines_ppo_20000_4", tensorboard_log="my_tf_board")
model.set_env(env)
model.learn(total_timesteps=30000, log_interval=30000, reset_num_timesteps=False)
model.save("nimble_quest_stable_baselines_ppo_20000_5")

# obs = env.reset()
# n_steps = 10000
# for step in range(n_steps):
#   action, _ = model.predict(obs, deterministic=True)
#   print("Step {}".format(step + 1))
#   print("Action: ", action)
#   obs, reward, done, info = env.step(action)
#   print('obs=', 'reward=', reward, 'done=', done)
#   if done:
#     # Note that the VecEnv resets automatically
#     # when a done signal is encountered
#     print("Goal reached!", "reward=", reward)
#     break