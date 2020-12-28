# import gym
# from power_switch import *
# import nq_gym
#
# env = gym.make('nq-v0')
# run_terminator_listener()
#
import os

os.system('python -m baselines.run --alg=ppo2 --env=nq_gym:nq-v0 --network=cnn '
          '--num_timesteps=300000 --save_path=openai_nq_weight --log_path=logs')

# --log_path=logs
# --network=cnn