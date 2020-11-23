from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("macOSX")

from tf_agents.environments import tf_py_environment
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.policies import policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from nq_environment import *
from power_switch import *


tf.compat.v1.enable_v2_behavior()


tempdir = "nimble_quest_weight_70"

num_iterations = 20  # @param {type:"integer"}
initial_collect_steps = 100  # @param {type:"integer"}
collect_steps_per_iteration = 1  # @param {type:"integer"}
replay_buffer_max_length = 1000  # @param {type:"integer"}
batch_size = 10  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
log_interval = 200  # @param {type:"integer"}
num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 10  # @param {type:"integer"}

run_terminator_listener()

nimble_quest_env = NQEnv()
nimble_quest_env = tf_py_environment.TFPyEnvironment(nimble_quest_env)
time_step = nimble_quest_env.reset()

action = np.array(5, dtype=np.int32)
next_time_step = nimble_quest_env.step(action)

# utils.validate_py_environment(nimble_quest_env, episodes=5)

fc_layer_params = (100, 100)
conv_layer_params = [(200, (8, 8), 4), (150, (4, 4), 2), (100, (3, 3), 1)]
dropout_layer = [0.1]

q_net = q_network.QNetwork(
    nimble_quest_env.observation_spec(),
    nimble_quest_env.action_spec(),
    # dropout_layer_params=dropout_layer,
    conv_layer_params=conv_layer_params,
    fc_layer_params=fc_layer_params)

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
global_step = tf.compat.v1.train.get_or_create_global_step()
# train_step_counter = tf.Variable(0)

#########################################################################
agent = dqn_agent.DqnAgent(
    nimble_quest_env.time_step_spec(),
    nimble_quest_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=global_step)

agent.initialize()
##########################################################################

eval_policy = agent.policy
collect_policy = agent.collect_policy

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=nimble_quest_env.batch_size,
    max_length=replay_buffer_max_length)

checkpoint_dir = os.path.join(tempdir, 'checkpoint')
train_checkpointer = common.Checkpointer(
    ckpt_dir=checkpoint_dir,
    max_to_keep=1,
    agent=agent,
    policy=agent.policy,
    replay_buffer=replay_buffer,
    global_step=global_step
)
train_checkpointer.initialize_or_restore()
# policy_dir = os.path.join(tempdir, 'policy')
tf_policy_saver = policy_saver.PolicySaver(agent.policy)

random_policy = random_tf_policy.RandomTFPolicy(nimble_quest_env.time_step_spec(), nimble_quest_env.action_spec())
random_policy.action(time_step)


def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


compute_avg_return(nimble_quest_env, random_policy, num_eval_episodes)


def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    # Add trajectory to the replay buffer
    buffer.add_batch(traj)


def collect_data(env, policy, buffer, steps):
    for _ in range(steps):
        collect_step(env, policy, buffer)


collect_data(nimble_quest_env, random_policy, replay_buffer, initial_collect_steps)
dataset = replay_buffer.as_dataset(num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2).prefetch(3)


############################################################

print(q_net.summary())
# print(q_net._q_value_layer.input_shape)
# print(q_net._q_value_layer.output_shape)
# print(agent)

iterator = iter(dataset)
print("itegrator" + str(iterator))
agent.train = common.function(agent.train)
agent.train_step_counter.assign(0)
train_checkpointer.initialize_or_restore()

print("################# running average_return once before training #########################")
avg_return = compute_avg_return(nimble_quest_env, agent.policy, num_eval_episodes) # Evaluate the agent's policy once before training.
returns = [avg_return]

print("################# training starting #########################")

for _ in range(num_iterations):
    # Collect a few steps using collect_policy and save to the replay buffer.
    collect_data(nimble_quest_env, agent.collect_policy, replay_buffer, collect_steps_per_iteration)
    # Sample a batch of data from the buffer and update the agent's network.
    experience, unused_info = next(iterator)
    train_loss = agent.train(experience).loss
    step = agent.train_step_counter.numpy()
    train_checkpointer.save(agent.train_step_counter)

    if step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss))

    if step % eval_interval == 0:
        avg_return = compute_avg_return(nimble_quest_env, agent.policy, num_eval_episodes)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
        returns.append(avg_return)

iterations = range(0, num_iterations + 1, eval_interval)
plt.plot(iterations, returns)
plt.ylabel('Average Return')
plt.xlabel('Iterations')
plt.ylim(top=250)


# def embed_mp4(filename):
#     """Embeds an mp4 file in the notebook."""
#     video = open(filename,'rb').read()
#     b64 = base64.b64encode(video)
#     tag = '''
#     <video width="640" height="480" controls>
#     <source src="data:video/mp4;base64,{0}" type="video/mp4">
#     Your browser does not support the video tag.
#     </video>'''.format(b64.decode())
#
#     return IPython.display.HTML(tag)


# def create_policy_eval_video(policy, filename, num_episodes=5, fps=30):
#     filename = filename + ".mp4"
#     with imageio.get_writer(filename, fps=fps) as video:
#         for _ in range(num_episodes):
#             inner_time_step = nimble_quest_env.reset()
#             video.append_data(nimble_quest_env.render())
#             while not inner_time_step.is_last():
#                 action_step = policy.action(inner_time_step)
#                 inner_time_step = nimble_quest_env.step(action_step.action)
#                 video.append_data(nimble_quest_env.render())
#     return embed_mp4(filename)


# saved_policy = tf.compat.v2.saved_model.load(policy_dir)
# create_policy_eval_video(saved_policy, "trained-agent")
# create_policy_eval_video(agent.policy, "trained-agent")

# create_policy_eval_video(random_policy, "random-agent")


