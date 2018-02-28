'''Base class for the DSRL paper toy game'''
import math

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from matplotlib import pyplot as plt

plt.ion()


class CrossCircleBase(gym.Env):
    '''Base class for DSRL paper cross-circle game'''
    metadata = {
        'render.modes': ['human'],
    }

    def __init__(self, field_dim=300):
        self.viewer = None
        self.field_dim = field_dim
        self.entity_size = 5

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0, 1, shape=(field_dim, field_dim))
        self.reward_range = (-1, 1)
        self.state = {'circle': np.zeros((self.field_dim, self.field_dim)),
                      'cross': np.zeros((self.field_dim, self.field_dim)),
                      'agent': np.zeros((self.field_dim, self.field_dim))
                     }
        self.entities = {}
        self.agent = {'center': None, 'top_left': None}

        self.seed()
        self.reset()

    def _step(self, action):
        # every move is 1/2 the agent's size
        action_type = ACTION_LOOKUP[action]
        step_size = self.entity_size/2 + self.entity_size % 2

        if action_type == 'UP':
            collision = self._move_agent(0, step_size)
        if action_type == 'DOWN':
            collision = self._move_agent(0, -step_size)
        if action_type == 'LEFT':
            collision = self._move_agent(-step_size, 0)
        if action_type == 'RIGHT':
            collision = self._move_agent(step_size, 0)

        if collision is not None:
            if collision == 'circle':
                reward = -1
            elif collision == 'cross':
                reward = 1
        else:
            reward = 0

        is_over = reward == -1
        info = {'entities': self.entities, 'agent': self.agent}
        combined_state = np.clip(np.add(
            np.add(self.state['circle'], self.state['cross']), self.state['agent']), 0, 1)
        return combined_state, reward, is_over, info

    def _reset(self):
        '''Should clear entities and state, call layout()'''
        raise NotImplementedError('Needs to be implemented in subclasses')

    def layout(self, random=True, mixed=True, num_entities=14):
        '''Sets up agent, crosses and circles on field. DOES NOT CLEAR FIELD.'''
        self.agent['center'] = np.array((self.field_dim/2, self.field_dim/2))
        self.agent['top_left'] = self.agent['center'] - self.entity_size/2
        self._draw_entity(self.agent['top_left'], 'agent')

        if not random:
            num_per_row = round(math.sqrt(num_entities))
            entity_spacing = (self.field_dim - num_per_row *
                              self.entity_size) / (num_per_row*2)

            def _get_next_coords():
                '''Generator for grid alignment'''
                last_grid_position = np.array([entity_spacing, entity_spacing])
                for _ in range(num_entities):
                    for __ in range(num_per_row):
                        last_grid_position += (entity_spacing +
                                               self.entity_size, 0)
                        # return center, top left
                        yield last_grid_position + self.entity_size/2, last_grid_position
                    # new row
                    last_grid_position = np.array(
                        [entity_spacing, last_grid_position[1]+entity_spacing+self.entity_size])

        for _ in range(num_entities):
            if mixed and num_entities % 2 == 0:
                entity_type = 'cross'
            else:
                entity_type = 'circle'

            shape = self._make_shape(entity_type)
            if random:
                center, top_left = self._get_random_entity_coords(shape)
            else:
                center, top_left = next(_get_next_coords())

            self._draw_entity(top_left, shape)
            self.entities[entity_type].append(
                {'center': center, 'top_left': top_left})

    def _render(self, mode='human', close=False):
        if close:
            plt.close()
        plt.imshow(self.state, interpolation='nearest')
        # plt.show()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _make_shape(self, entity_type):
        '''Types: circle/cross/agent'''
        if self.entity_size != 5:
            raise Exception('Only entity size supported right now is 5x5')

        if entity_type == 'circle':
            return np.array([
                [0, 1, 1, 1, 0],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [0, 1, 1, 1, 0]
            ])
        elif entity_type == 'cross':
            return np.array([
                [1, 0, 0, 0, 1],
                [0, 1, 0, 1, 0],
                [0, 0, 1, 0, 0],
                [0, 1, 0, 1, 0],
                [1, 0, 0, 0, 1]
            ])
        elif entity_type == 'agent':
            return np.array([
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [1, 1, 1, 1, 1],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0]
            ])

    def _draw_entity(self, top_left, entity_type):
        window = self._get_state_window(top_left)
        window[entity_type][:, :] = np.clip(
            np.add(window, self._make_shape(entity_type)), 0, 1)

    def _undraw_entity(self, top_left, entity_type):
        window = self._get_state_window(top_left)
        window[entity_type][:, :] = np.subtract(
            window, self._make_shape(entity_type))

    def _get_random_entity_coords(self, shape):
        '''Returns tuple of ([center_x, center_y], [top_left_x, top_left_y])'''
        center = self.np_random.randint(
            self.entity_size/2, self.field_dim-self.entity_size/2, (2,))
        top_left = center - self.entity_size/2

        while self._detect_collision(top_left, shape) is None:
            # spot's taken, find a new one
            center, top_left = self._get_random_entity_coords(shape)

        return center, top_left

    def _detect_collision(self, top_left, shape):
        window = self._get_state_window(top_left)
        # check across all layers
        for layer, layer_state in window.items():
            sum_map = np.add(layer_state, shape)
            for sum_x, sum_y in np.ndenumerate(sum_map):
                # if there's a 2, there are 2 1's on top of each other = collision
                if sum_map[sum_x, sum_y] == 2:
                    return layer, np.array((top_left[0] + sum_x, top_left[1] + sum_y))
        return None

    def _get_state_window(self, top_left, size=None):
        if size is None:
            size = self.entity_size

        return dict([
            (layer, (self.state[layer][top_left[0]:top_left[0]+size,
                                       top_left[1]:top_left[1]+size])
            )
            for layer in ['circle', 'cross', 'agent']
        ])

    def _move_agent(self, x_step, y_step):
        '''Returns type of colliding object if collided'''
        new_agent_x = self.agent['top_left'][0] + x_step
        new_agent_y = self.agent['top_left'][1] + y_step

        # If collides with wall, do not move
        if (not 0 <= new_agent_x <= self.field_dim - self.entity_size or
                not 0 <= new_agent_y <= self.field_dim - self.entity_size):
            return

        new_top_left = (new_agent_x, new_agent_y)
        # Remove current agent representation
        self._undraw_entity(new_top_left, 'agent')

        collision_type, collision_pixel_coords = self._detect_collision(
            new_top_left, self._make_shape('agent'))
        collision_top_left = self._find_entity_by_pixel(
            collision_pixel_coords, collision_type)
        self._undraw_entity(collision_top_left, collision_type)

        self._draw_entity(new_top_left, 'agent')

    def _find_entity_by_pixel(self, pixel_coords, entity_type):
        '''Match shape in window so that pixel_coords is in the shape'''
        layer_window = self._get_state_window(
            pixel_coords-(self.entity_size, self.entity_size), 2*self.entity_size)[entity_type]
        for top_left_x, top_left_y in np.ndenumerate(layer_window):
            # try to match shape against window that originates in top_left_x, top_left_y
            sum_map = np.add(layer_window[top_left_x:top_left_x+5,
                                          top_left_y:top_left_y+5], self._make_shape(entity_type))
            if 1 in sum_map[top_left_x:top_left_x+5, top_left_y:top_left_y+5]:
                continue   # not full match of shape
            elif sum_map[self.entity_size, self.entity_size] == 2:
                # original pixel coords are at the center, i.e. (self.entity_size, self.entity_size)
                # If they're matched, shape is found
                # Return global coordinates of matched top left coordinates
                return np.add((top_left_x, top_left_y),
                              pixel_coords-(self.entity_size, self.entity_size))
        raise Exception('Unmatched collision. You should never see this message.\n\
                         Coords: {}, shape: {}'.format(pixel_coords, entity_type))


ACTION_LOOKUP = {
    0: 'UP',
    1: 'RIGHT',
    2: 'DOWN',
    3: 'LEFT'
}
