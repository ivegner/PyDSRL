from gym.envs.registration import register

register(
    id='CrossCircle-NegGrid-v0',
    entry_point='cross_circle_gym.envs:CrossCircleNegGrid',
)
register(
    id='CrossCircle-MixedGrid-v0',
    entry_point='cross_circle_gym.envs:CrossCircleMixedGrid',
)
register(
    id='CrossCircle-NegRand-v0',
    entry_point='cross_circle_gym.envs:CrossCircleNegRand',
)
register(
    id='CrossCircle-MixedRand-v0',
    entry_point='cross_circle_gym.envs:CrossCircleMixedRand',
)