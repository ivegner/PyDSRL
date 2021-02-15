'''Cross-circle game with mixed random rewards'''
import numpy as np
from cross_circle_gym.envs.cross_circle_base import CrossCircleBase

class CrossCircleMixedRand(CrossCircleBase):
    '''Environment providing a grid of circles (negative rewards)'''

    def setup_field(self):
        self.layout(random=True, mixed=True)

    def make_random_state(self, min_entities=1, max_entities=30):
        '''Make random layout for training, return state'''
        self.entities = {'cross': [], 'circle': []}
        self.agent = {'center': None, 'top_left': None}
        self.state = {'circle': np.zeros((self.field_dim, self.field_dim)),
                      'cross': np.zeros((self.field_dim, self.field_dim)),
                      'agent': np.zeros((self.field_dim, self.field_dim))
                     }
        self.layout(random=True,
                    mixed=True,
                    min_entities=min_entities,
                    max_entities=max_entities,
                    random_agent=True)
        return self.combined_state
