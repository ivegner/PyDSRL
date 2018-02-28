from cross_circle_gym.envs.cross_circle_base import CrossCircleBase

class CrossCircleMixedGrid(CrossCircleBase):
    '''Environment providing a grid of circles (negative rewards)'''

    def setup_field(self):
        self.layout(random=False, mixed=True)