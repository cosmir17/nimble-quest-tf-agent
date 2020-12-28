from gym.envs.registration import register

register(
    id='nq-v0',
    entry_point='nq_gym.envs:NQEnv',
)
