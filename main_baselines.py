from stable_baselines3 import PPO
# from stable_baselines3 import A2C
from stable_baselines3.common.cmd_util import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3.a2c.policies import CnnPolicy
from stable_baselines3.ppo.policies import CnnPolicy
from stable_baselines3.common.env_checker import check_env
from nq_environment_baselines import *
from power_switch import *

run_terminator_listener()

# Instantiate the env
env = DummyVecEnv([lambda: NQEnv()])
# Train the agent
model = PPO(CnnPolicy, env, verbose=2).learn(total_timesteps=20000, log_interval=1)
model.save("nimble_quest_stable_baselines_ppo_20000")

# model = PPO.load("nimble_quest_stable_baselines_ppo_20000")
# model.learn(total_timesteps=20000, log_interval=1)
# model.set_env(env)

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
# print("model trained, mean_reward : " + str(mean_reward) + " std_reward: " + str(std_reward))

# obs = env.reset()
# n_steps = 100
# for step in range(n_steps):
#   action, _ = model.predict(obs, deterministic=True)
#   print("Step {}".format(step + 1))
#   print("Action: ", action)
#   obs, reward, done, info = env.step(action)
#   print('obs=', obs, 'reward=', reward, 'done=', done)
#   if done:
#     # Note that the VecEnv resets automatically
#     # when a done signal is encountered
#     print("Goal reached!", "reward=", reward)
#     break