# import gym
# from power_switch import *
# import nq_gym
#
# env = gym.make('nq-v0')
# run_terminator_listener()

from baselines.run import *
from power_switch import run_terminator_listener

run_terminator_listener()

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     tf.config.experimental.set_virtual_device_configuration(gpus[0],
#             [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000)])
#     print("****************** setting memory limit 5000mb ******************")
#   except RuntimeError as e:
#     print(e)

# tf.compat.v1.disable_eager_execution()

save_args = ["--alg", "ppo2", "--env", "nq_gym:nq-v0", "--network", "cnn",
            "--num_timesteps", "500000", "--save_path", "openai_nq_weight", "--log_path", "logs"]

load_args = ["--alg", "ppo2", "--env", "nq_gym:nq-v0", "--network", "cnn",
            "--num_timesteps", "0", "--load_path", "openai_nq_weight", "--log_path", "logs",
            "--play", ""]

main(save_args)


# --log_path=logs
# --network=cnn