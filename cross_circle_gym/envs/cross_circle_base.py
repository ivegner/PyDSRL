'''Base class for the DSRL paper toy game'''
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
import imageio
import os
from skimage.transform import resize


class Entity(object):
    def __init__(self, y, x, h, w, kind, center=False, z=None):
        if center:
            self.top = y - h / 2
            self.left = x - w / 2
        else:
            self.top = y
            self.left = x

        self.h = h
        self.w = w
        self.alive = True
        self.z = np.random.rand() if z is None else z
        self.kind = kind

    @property
    def right(self):
        return self.left + self.w

    @property
    def bottom(self):
        return self.top + self.h

    def intersects(self, r2):
        return self.overlap_area(r2) > 0

    def overlap_area(self, r2):
        overlap_bottom = np.minimum(self.bottom, r2.bottom)
        overlap_top = np.maximum(self.top, r2.top)

        overlap_right = np.minimum(self.right, r2.right)
        overlap_left = np.maximum(self.left, r2.left)

        area = np.maximum(overlap_bottom - overlap_top, 0) * np.maximum(overlap_right - overlap_left, 0)
        return area

    def centre(self):
        return (
            self.top + self.h / 2.,
            self.left + self.w / 2.
        )

    def __str__(self):
        return "<{}:{} {}:{}, alive={}, z={}, kind={}>".format(self.top, self.bottom, self.left, self.right, self.alive, self.z, self.kind)

    def __repr__(self):
        return str(self)


class CrossCircleBase(gym.Env):
    '''Base class for DSRL paper cross-circle game'''
    metadata = {
        'render.modes': ['human'],
    }

    def __init__(
            self, field_dim=100, background_colour='white', shape_colours="white white white",
            entity_size=10, min_entities=25, max_entities=50, max_overlap_factor=0.2, overlap_factor=0.25, step_size=10):

        self.field_dim = field_dim
        self.background_colour = background_colour
        self.shape_colours = shape_colours
        self.entity_size = entity_size

        self.min_entities = min_entities
        self.max_entities = max_entities

        self.max_overlap_factor = max_overlap_factor
        self.overlap_factor = overlap_factor
        self.step_size = step_size

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0, 1, shape=(self.field_dim, self.field_dim, 3))
        self.reward_range = (-1, 1)

        self.entities = {'cross': [], 'circle': []}
        self.agent = None

        self.masks = {}
        for entity_type in 'circle cross agent'.split():
            f = os.path.join(os.path.dirname(__file__), "images", "{}.png".format(entity_type))
            mask = imageio.imread(f)
            mask = resize(mask, (self.entity_size, self.entity_size), mode='edge', preserve_range=True)
            self.masks[entity_type] = np.tile(mask[..., 3:], (1, 1, 3)) / 255.

        self.background_colour = None
        if background_colour:
            colour = to_rgb(background_colour)
            colour = np.array(colour)[None, None, :]
            self.background_colour = colour

        self.shape_colours = None
        if shape_colours:
            if isinstance(shape_colours, str):
                shape_colours = shape_colours.split()

            self.shape_colours = {
                entity_type: np.array(to_rgb(c))[None, None, :]
                for entity_type, c in zip(sorted("agent circle cross".split()), shape_colours)}

        self.seed()
        self.reset()
        self.viewer = None

    @property
    def combined_state(self):
        '''Add state layers into one array'''
        image = np.zeros((self.field_dim, self.field_dim, 3)) * self.background_colour

        all_entities = []
        for entity_type, entities in self.entities.items():
            all_entities.extend(entities)
        all_entities.append(self.agent)

        all_entities = sorted(all_entities, key=lambda x: x.z)

        for entity in all_entities:
            if not entity.alive:
                continue

            _alpha = self.masks[entity.kind]
            if self.shape_colours is None:
                _image = np.random.rand(self.entity_size, self.entity_size, 3)
            else:
                _image = np.tile(self.shape_colours[entity.kind], (self.entity_size, self.entity_size, 1))

            top = int(entity.top)
            bottom = top + int(entity.h)

            left = int(entity.left)
            right = left + int(entity.w)

            image[top:bottom, left:right, ...] = _alpha * _image + (1 - _alpha) * image[top:bottom, left:right, ...]

        return image

    def step(self, action):
        action_type = ACTION_LOOKUP[action]
        step_size = self.step_size if self.step_size is not None else (self.entity_size/2 + self.entity_size % 2)

        if action_type == 'UP':
            collision = self._move_agent(0, step_size)
        if action_type == 'DOWN':
            collision = self._move_agent(0, -step_size)
        if action_type == 'LEFT':
            collision = self._move_agent(-step_size, 0)
        if action_type == 'RIGHT':
            collision = self._move_agent(step_size, 0)

        reward = collision['cross'] - collision['circle']

        info = {'entities': self.entities, 'agent': self.agent}
        return self.combined_state, reward, False, info

    def reset(self):
        '''Clear entities and state, call setup_field()'''
        self.entities = {'cross': [], 'circle': []}
        self.agent = None
        self.setup_field()
        return self.combined_state

    def setup_field(self):
        '''Calls layout. Meant as a chance for subclasses to alter layout() call'''
        raise NotImplementedError('Needs to be implemented in subclasses')

    def layout(self, random=True, mixed=True, min_entities=None, max_entities=None, random_agent=False):
        '''Sets up agent, crosses and circles on field. DOES NOT CLEAR FIELD.'''
        if min_entities is None:
            min_entities = self.min_entities
        if max_entities is None:
            max_entities = self.max_entities

        n_entities = np.random.randint(min_entities, max_entities+1)

        if random_agent:
            top_left = np.random.randint(self.field_dim-self.entity_size, size=(2,))
            self.agent = Entity(top_left[0], top_left[1], self.entity_size, self.entity_size, kind="agent")
        else:
            self.agent = Entity(
                self.field_dim/2, self.field_dim/2, self.entity_size, self.entity_size, kind="agent", center=True)

        if random:
            sub_image_shapes = [(self.entity_size, self.entity_size) for i in range(n_entities)]
            entities = self._sample_entities(sub_image_shapes, self.max_overlap_factor)

            for i, e in enumerate(entities):
                if mixed and i % 2 == 0:
                    entity_type = 'cross'
                else:
                    entity_type = 'circle'
                e.kind = entity_type
                self.entities[entity_type].append(e)
        else:
            n_per_row = int(round(np.sqrt(n_entities)))
            n_entities = n_per_row ** 2
            center_spacing = self.field_dim / n_per_row
            row_start_entity_type = 'circle'

            for i in range(n_per_row):
                entity_type = row_start_entity_type
                for j in range(n_per_row):
                    y = center_spacing / 2 + center_spacing * i
                    x = center_spacing / 2 + center_spacing * j

                    entity = Entity(y, x, self.entity_size, self.entity_size, center=True, kind=entity_type)

                    self.entities[entity_type].append(entity)

                    if mixed:
                        entity_type = 'cross' if entity_type == 'circle' else 'circle'

                if mixed:
                    row_start_entity_type = 'cross' if row_start_entity_type == 'circle' else 'circle'

        # Clear objects that overlap with the agent originally
        self._move_agent(0, 0)

    def _sample_entities(self, patch_shapes, max_overlap_factor=None, size_std=None):
        if len(patch_shapes) == 0:
            return []

        patch_shapes = np.array(patch_shapes)
        n_rects = patch_shapes.shape[0]

        rects = []

        for i in range(n_rects):
            n_tries = 0
            while True:
                if size_std is None:
                    shape_multipliers = 1.
                else:
                    shape_multipliers = np.maximum(np.random.randn(2) * size_std + 1.0, 0.5)

                m, n = np.ceil(shape_multipliers * patch_shapes[i, :2]).astype('i')

                rect = Entity(
                    np.random.randint(0, self.field_dim-m+1),
                    np.random.randint(0, self.field_dim-n+1), m, n, kind=None)

                if max_overlap_factor is None:
                    rects.append(rect)
                    break
                else:
                    violation = False
                    for r in rects:
                        if rect.overlap_area(r) / (self.entity_size**2) > max_overlap_factor:
                            violation = True
                            break

                    if not violation:
                        rects.append(rect)
                        break

                n_tries += 1

                if n_tries > 10000:
                    raise Exception(
                        "Could not fit rectangles. "
                        "(n_rects: {}, field_dim: {}, max_overlap_factor: {})".format(
                            n_rects, self.field_dim, max_overlap_factor))

        return rects

    def render(self, mode='human', close=False, wait=0):
        plt.ion()
        if self.viewer is None:
            self.viewer = plt.imshow(self.combined_state)
        self.viewer.set_data(self.combined_state)
        plt.pause(wait)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _make_shape(self, entity_type):
        return self.masks[entity_type]

    def _move_agent(self, x_step, y_step):
        agent = self.agent
        collisions = {entity_type: 0 for entity_type in self.entities}

        for entity_type, entities in self.entities.items():
            for i, entity in enumerate(entities):
                if entity.alive and agent.overlap_area(entity) > self.overlap_factor * (self.entity_size ** 2):
                    collisions[entity_type] += 1
                    entity.alive = False

        n_segments = 10

        if not x_step and not y_step:
            return collisions

        for i in range(n_segments):
            new_x = agent.left + x_step / n_segments
            new_y = agent.top + y_step / n_segments

            wall_collision = (
                new_x + agent.w > self.field_dim or
                new_x < 0 or
                new_y + agent.h > self.field_dim or
                new_y < 0
            )

            if wall_collision:
                break
            else:
                agent.left = new_x
                agent.top = new_y

                for entity_type, entities in self.entities.items():
                    for i, entity in enumerate(entities):
                        if entity.alive and agent.overlap_area(entity) > self.overlap_factor * (self.entity_size ** 2):
                            collisions[entity_type] += 1
                            entity.alive = False

        return collisions


ACTION_LOOKUP = {
    0: 'UP',
    1: 'RIGHT',
    2: 'DOWN',
    3: 'LEFT'
}
