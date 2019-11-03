from gym.envs.registration import register

register(
    id='Pushing2D-v1',
    entry_point='envs.2Dpusher_env:Pusher2d',
    kwargs={'control_noise': 0}
)

# Control noise is added to the action
register(
    id='Pushing2DNoisyControl-v1',
    entry_point='envs.2Dpusher_env:Pusher2d',
    kwargs={'control_noise': 0.3}
)
