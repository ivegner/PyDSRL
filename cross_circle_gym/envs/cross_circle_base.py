'''Base class for the DSRL paper toy game'''
import math

import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from numpy.lib.stride_tricks import as_strided
from matplotlib import pyplot as plt

plt.ion()

class CrossCircleBase(gym.Env):
    '''Base class for DSRL paper cross-circle game'''
    metadata = {
        'render.modes': ['human'],
    }

    def __init__(self, field_dim=50):
        self.field_dim = field_dim
        self.entity_size = 5
        self.render_wait = 1

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0, 1, shape=(field_dim, field_dim), dtype='f')
        self.reward_range = (-1, 1)
        self.state = {'circle': np.zeros((self.field_dim, self.field_dim)),
                      'cross': np.zeros((self.field_dim, self.field_dim)),
                      'agent': np.zeros((self.field_dim, self.field_dim))
                     }
        self.entities = {'cross': [], 'circle': []}
        self.agent = {'center': None, 'top_left': None}

        self.seed()
        self.reset()
        self.viewer = plt.imshow(self.combined_state)

    @property
    def combined_state(self):
        '''Add state layers into one array'''
        return np.clip(
            np.add(np.add(self.state['circle'], self.state['cross']), self.state['agent']), 0, 1)

    def step(self, action):

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
        return self.combined_state, reward, is_over, info

    def reset(self):
        '''Clear entities and state, call setup_field()'''
        self.entities = {'cross': [], 'circle': []}
        self.agent = {'center': None, 'top_left': None}
        self.state = {'circle': np.zeros((self.field_dim, self.field_dim)),
                      'cross': np.zeros((self.field_dim, self.field_dim)),
                      'agent': np.zeros((self.field_dim, self.field_dim))
                     }
        self.setup_field()

    def setup_field(self):
        '''Calls layout. Meant as a chance for subclasses to alter layout() call'''
        raise NotImplementedError('Needs to be implemented in subclasses')

    def layout(self, random=True, mixed=True, num_entities=16):
        '''Sets up agent, crosses and circles on field. DOES NOT CLEAR FIELD.'''
        self.agent['center'] = self._round_int_ndarray((self.field_dim/2, self.field_dim/2))
        self.agent['top_left'] = self._round_int_ndarray(self.agent['center'] - self.entity_size/2)
        self._draw_entity(self.agent['top_left'], 'agent')

        if not random:
            num_per_row = round(math.sqrt(num_entities))
            entity_spacing = (self.field_dim/num_per_row - self.entity_size)/2
            def _get_next_coords(mixed):
                '''Generator for grid alignment'''
                last_grid_position = np.array([entity_spacing, entity_spacing])
                entity_type = 'circle'
                for _ in range(num_entities):
                    for __ in range(num_per_row):
                        # return center, top left
                        if mixed:
                            entity_type = 'cross' if entity_type == 'circle' else 'circle'
                        yield self._round_int_ndarray(last_grid_position + self.entity_size/2), \
                              self._round_int_ndarray(last_grid_position), entity_type

                        last_grid_position += (0, 2*entity_spacing + self.entity_size)
                    # new row
                    last_grid_position[:] = [last_grid_position[0]+2*entity_spacing+self.entity_size, entity_spacing]
                    entity_type = 'cross' if entity_type == 'circle' else 'circle'  # checkerboard
            grid_gen = _get_next_coords(mixed)

        for idx in range(num_entities):
            if random:
                if mixed and idx % 2 == 0:
                    entity_type = 'cross'
                else:
                    entity_type = 'circle'
                center, top_left = self._get_random_entity_coords(entity_type)
            else:
                center, top_left, entity_type = next(grid_gen)
            self._draw_entity(top_left, entity_type)
            self.entities[entity_type].append(
                {'center': center, 'top_left': top_left})

    def render(self, mode='human'):
        self.viewer.set_data(self.combined_state)
        plt.pause(self.render_wait)

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
        window = self._get_state_window(top_left)[entity_type]
        scene = np.add(window, self._make_shape(entity_type))
        window[:, :] = scene

    def _undraw_entity(self, top_left, entity_type):
        window = self._get_state_window(top_left)[entity_type]
        window[:, :] = np.subtract(
            window, self._make_shape(entity_type))


    def _get_random_entity_coords(self, entity_type):
        '''Returns tuple of ([center_x, center_y], [top_left_x, top_left_y])'''
        shape = self._make_shape(entity_type)

        center = self.np_random.randint(
            self.entity_size/2, self.field_dim-self.entity_size/2, (2,))
        top_left = self._round_int_ndarray(center - self.entity_size/2)

        if self._detect_collision(top_left, shape) is not None:
            # spot's taken, find a new one
            center, top_left = self._get_random_entity_coords(entity_type)

        return center, top_left

    def _detect_collision(self, top_left, shape):
        window = self._get_state_window(top_left)
        # check across all layers
        for layer, layer_state in window.items():
            sum_map = np.add(layer_state, shape).astype(int)
            for (sum_x, sum_y), val in np.ndenumerate(sum_map):
                # if there's a 2, there are 2 1's on top of each other = collision
                if val == 2:
                    return layer, self._round_int_ndarray((top_left[0] + sum_x, top_left[1] + sum_y))
        return None

    def _get_state_window(self, top_left, size=None):
        if size is None:
            size = self.entity_size

        window = {}
        for layer in ['circle', 'cross', 'agent']:
            window[layer] = self.state[layer][top_left[0]:top_left[0]+size,
                                              top_left[1]:top_left[1]+size]
        return window

    def _move_agent(self, x_step, y_step):
        '''Returns type of colliding object if collided'''
        new_agent_x = self.agent['top_left'][0] + x_step
        new_agent_y = self.agent['top_left'][1] + y_step
        new_agent_center = self._round_int_ndarray(self.agent['center'] + (x_step, y_step))

        # If collides with wall, do not move
        if (not 0 <= new_agent_x <= self.field_dim - self.entity_size or
                not 0 <= new_agent_y <= self.field_dim - self.entity_size):
            logger.debug('agent hit wall')
            return

        new_top_left = self._round_int_ndarray((new_agent_x, new_agent_y))
        # Remove current agent representation
        self._undraw_entity(self.agent['top_left'], 'agent')

        collision = self._detect_collision(
            new_top_left, self._make_shape('agent'))
        if collision is not None:
            collision_type, collision_pixel_coords = collision
            collision_top_left = self._find_entity_by_pixel(
                collision_pixel_coords, collision_type)
            self._undraw_entity(collision_top_left, collision_type)

        self._draw_entity(new_top_left, 'agent')
        self.agent = {'top_left': new_top_left, 'center': new_agent_center}
        return collision[0] if collision is not None else None

    def _find_entity_by_pixel(self, pixel_coords, entity_type):
        '''Match shape in window so that pixel_coords is in the shape'''
        window_top_left = pixel_coords-(self.entity_size, self.entity_size)
        off_the_edge = window_top_left[np.where(window_top_left < 0)]
        if off_the_edge:
            off_the_edge = abs(np.min(off_the_edge))
            window_top_left += off_the_edge
            # print(window_top_left, self._get_state_window(window_top_left, 2*self.entity_size)[entity_type])
        else:
            off_the_edge = 0
        layer_window = self._get_state_window(window_top_left, 2*self.entity_size)[entity_type]

        # if not layer_window.any():
        #     plt.savefig('unmatched_debug.png')
        #     print('pixel coords: ', pixel_coords)
        #     print('self._get_state_window({}, {})[{}]'.format(pixel_coords-(self.entity_size, self.entity_size),
        #                                                       2*self.entity_size, entity_type))
        views = conv2d(layer_window, (self.entity_size, self.entity_size))
        for top_left_x, row in enumerate(views):
            for top_left_y, view in enumerate(row):
                # try to match shape against window that originates in top_left_x, top_left_y]
                shape = self._make_shape(entity_type)
                sum_map = np.add(view, shape)
                for coords, val in np.ndenumerate(shape):
                    if val == 1 and sum_map[coords] == 1:
                        # not full match of shape because not superimposed fully
                        break
                else:
                    if sum_map[(self.entity_size-top_left_x-off_the_edge, self.entity_size-top_left_y-off_the_edge)] == 2:
                        # original pixel coords are at the center, i.e. (self.entity_size, self.entity_size)
                        # If they're matched, shape is found
                        # Return global coordinates of matched top left coordinates
                        return np.add((top_left_x, top_left_y), window_top_left, dtype=np.int32)
        plt.savefig('unmatched_debug.png')
        raise Exception('Unmatched collision. You should never see this message.\n\
                         Coords: {}, shape: {}'.format(pixel_coords, entity_type))

    def _round_int_ndarray(self, array_like):
        return np.around(array_like).astype(int)

def conv2d(arr, sub_shape):
    '''convolve sub_shape over arr'''
    view_shape = tuple(np.subtract(arr.shape, sub_shape)+1) + sub_shape
    try:
        arr_view = as_strided(arr, (view_shape), arr.strides * 2, writeable=False)
    except Exception as exc:
        print('arr: ', arr, 'sub: ', sub_shape)
        print(exc)
    # arr_view = arr_view.reshape((-1,) + sub_shape)

    return arr_view


ACTION_LOOKUP = {
    0: 'UP',
    1: 'RIGHT',
    2: 'DOWN',
    3: 'LEFT'
}
