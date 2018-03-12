'''Build full state representation'''
import copy

import numpy as np
from pandas import DataFrame

from .autoencoder import NEIGHBOR_RADIUS


class StateRepresentationBuilder():
    '''
    Implements section 3.2 of the paper. Builds state representation \
    of raw entities
    '''
    def __init__(self):
        self.tracked_entities = []
        self.next_free_entity_id = 0
        self.type_transition_matx = None
        self.total_transitions = []
        self.do_not_exist = []  # entities to be removed as they no longer exist
        self.sim_weights = [1/3, 1/3, 1/3]

    def build_state(self, entities, found_types):
        '''Tag entities across time, build interactions'''
        if not self.tracked_entities:
            # Init type transition matrix
            self.type_transition_matx = DataFrame(0,
                                        columns=['null'] + found_types,
                                        index=['null'] + found_types)

            # init tracking for objects
            self._init_tracking(entities)

        else:
            # Update type transition matrix if there are new types
            num_current_types = self.type_transition_matx.shape()[0]
            for e_type in found_types:
                if e_type not in self.type_transition_matx.index:
                    # New, never before seen entity type, make new entry in trans matrix
                    # make column
                    self.type_transition_matx.insert(num_current_types, e_type, 0)
                    # make row
                    self.type_transition_matx.loc[e_type] = np.zeros(num_current_types, dtype=int)

            # Update tracking for objects
            self._update_tracking(entities)

        final_repr = self._build_representation()

        return final_repr

    def _init_tracking(self, entities):
        '''Set up tags for all existing entities'''
        for entity in entities:
            entity.id = self.next_free_entity_id
            self.tracked_entities.append(entity)
            self.next_free_entity_id += 1

    def _is_same_entity(self, old_e, new_e):
        '''Check whether new_e is a displaced version of old_e'''
        similarity = 0

        # Factor 1: euclidean distance
        l_dist = 1 / (1 + np.linalg.norm(new_e.position-old_e.position))
        # Factor 2: Transitions
        l_trans = self.type_transition_matx[old_e.type, new_e.type] / \
                  self.total_transitions[old_e.type]
        # factor 3: neighbors
        l_neighbors = 1 / (1 + abs(new_e.n_neighbors - old_e.n_neighbors))

        similarity = self.sim_weights[0] * l_dist + \
                     self.sim_weights[1] * l_trans + \
                     self.sim_weights[2] * l_neighbors

        print('Similarity: ', similarity)
        return similarity > 0.5

    def _update_tracking(self, new_entities):
        '''Track entities across time, using their last state'''
        self.prev_tracked_entities = copy.deepcopy(self.tracked_entities)
        new_entities = []

        # if an entity is not matched with any in new entities,
        # place it in possibly_disappeared, and remove it if encountered
        # If there are any in possibly_disappeared by the time the
        # iteration over new_entities is done, the entity has actually disappeared
        possibly_disappeared = []

        newly_nonexistent = []
        for i, tracked_e in enumerate(self.tracked_entities):
            if not tracked_e.exists:
                newly_nonexistent.append(i)
                continue
            for new_e_i, new_e in enumerate(new_entities):
                if self._is_same_entity(tracked_e, new_e):
                    # Update transition matrix
                    # (even if not transitioned, how often the type stays the same is important)
                    self._mark_transition(tracked_e.type, new_e.type)

                    self.tracked_entities[i].update_self(new_e)
                    if i in possibly_disappeared:
                        possibly_disappeared.remove(i)
                    del new_entities[new_e_i]
                    break
            else:
                # new entity, and/or tracked_e disappeared
                possibly_disappeared.append(i)

        for disapp_idx in possibly_disappeared: # well, they definitely disappeared
            # Mark transition in matrix
            self._mark_transition(self.tracked_entities[disapp_idx].type, 'null')

            # Mark object as nonexistent, to be removed in the next timestep
            # We do not delete it in this timestep because we have to show the agent
            # that the entity disappeared
            self.tracked_entities[disapp_idx].disappeared()

        self.do_not_exist.reverse()
        for dne_idx in self.do_not_exist:
            del self.tracked_entities[dne_idx]

        self.do_not_exist = newly_nonexistent   # to be removed next time

        for entity_to_add in new_entities:
            entity_to_add.id = self.next_free_entity_id
            entity_to_add.appeared()
            self.tracked_entities.append(entity_to_add)

            # Mark transition in matrix
            self._mark_transition('null', entity_to_add.type)

            # increment id for next appearing object
            self.next_free_entity_id += 1


    def _build_representation(self):
        '''Build time-abstracted representation + object interactions'''

        def interaction(e1, e2, loc_diff, types_before, types_after):
            '''Make interaction dict for certain interaction params'''
            return {
                'interaction': (e1.id, e2.id),
                'loc_difference': loc_diff,
                'types_before': types_before,
                'types_after': types_after
            }
        interactions = []
        interactions_built = [] # pairs of entities for which interaction has already been built
        # Build interactions
        for entity in self.tracked_entities:
            within_radius = [x for x in self.tracked_entities
                               if np.all((x.position - entity.position) < NEIGHBOR_RADIUS)]
            for w_r in within_radius:
                interact_code = (entity.id, w_r.id) if entity.id < w_r.id else (w_r.id, entity.id)
                if interact_code in interactions_built:
                    #interaction already built before
                    continue

                # position change
                loc_diff = (entity.position - entity.prev_state['position']) - \
                                    (w_r.position - w_r.prev_state['position'])
                types_before = (entity.prev_state['type'], w_r.prev_state['type'])
                types_after = (entity.entity_type, w_r.entity_type)
                interactions.append(interaction(entity, w_r, loc_diff, types_before, types_after))
                interactions_built.append(interact_code)

        return interactions

    def _mark_transition(self, from_type, to_type):
        '''Mark type transition in transition matrix'''
        self.total_transitions[from_type] += 1
        self.type_transition_matx.loc[from_type, to_type] += 1
