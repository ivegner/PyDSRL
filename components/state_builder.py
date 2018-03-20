'''Build full state representation'''
import numpy as np
from pandas import DataFrame

class StateRepresentationBuilder():
    '''
    Implements section 3.2 of the paper. Builds state representation \
    of raw entities
    '''
    def __init__(self, neighbor_radius=25):
        self.tracked_entities = []
        self.next_free_entity_id = 0
        self.type_transition_matx = None
        self.total_transitions = {}
        self.do_not_exist = []  # entities to be removed as they no longer exist
        self.sim_weights = [2, 1, 1]
        self.neighbor_radius = neighbor_radius

    def build_state(self, entities, found_types):
        '''Tag entities across time, build interactions'''
        if not self.tracked_entities and self.next_free_entity_id == 0:
            # Init type transition matrix
            self.type_transition_matx = DataFrame(0,
                                                  columns=['null'] + found_types,
                                                  index=['null'] + found_types)

            for e_type in found_types:
                # set initial transition to 0 because assumption: objects tend to stay the same
                self.type_transition_matx.at[e_type, e_type] = 1

            # init tracking for objects
            self._init_tracking(entities)

        else:
            # Update type transition matrix if there are new types
            num_current_types = self.type_transition_matx.shape[0]
            for e_type in found_types:
                if e_type not in self.type_transition_matx.index:
                    # New, never before seen entity type, make new entry in trans matrix
                    # make column
                    self.type_transition_matx.insert(num_current_types, e_type, 0)
                    # make row
                    self.type_transition_matx.loc[e_type] = np.zeros(num_current_types, dtype=int)

                    # set initial transition to 0 because assumption: objects tend to stay the same
                    self.type_transition_matx.at[e_type, e_type] = 1

            # print(self.type_transition_matx)

            # Update tracking for objects
            self._update_tracking(entities)

        final_repr = self._build_representation()

        return final_repr

    def restart(self):
        '''Fresh start of a new episode'''
        self.tracked_entities = []
        self.next_free_entity_id = 0
        self.do_not_exist = []

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
        l_trans = self.type_transition_matx.at[old_e.entity_type, new_e.entity_type] / \
                  self.total_transitions.setdefault(old_e.entity_type, 0)
        if np.isnan(l_trans):
            l_trans = 0
        # factor 3: neighbors
        l_neighbors = 1 / (1 + abs(new_e.n_neighbors - old_e.n_neighbors))

        # print('----')
        # print('before:', old_e.__dict__)
        # print('new:', new_e.__dict__)
        similarity = self.sim_weights[0] * l_dist + \
                     self.sim_weights[1] * l_trans + \
                     self.sim_weights[2] * l_neighbors
        similarity = similarity/3

        # print(l_dist, l_trans, l_neighbors, 'similarity:', similarity)

        return similarity > 0.5

    def _update_tracking(self, new_entities):
        '''Track entities across time, using their last state'''

        # if an entity is not matched with any in new entities,
        # place it in possibly_disappeared, and remove it if encountered
        # If there are any in possibly_disappeared by the time the
        # iteration over new_entities is done, the entity has actually disappeared
        possibly_disappeared = []

        newly_nonexistent = []
        for i, tracked_e in enumerate(self.tracked_entities):
            # print(tracked_e.__dict__)
            if not tracked_e.exists:
                print('Marked for deletion next loop', tracked_e.__dict__)
                print('---')
                newly_nonexistent.append(i)
                continue
            for new_e_i, new_e in enumerate(new_entities):
                # print('comparing', new_e.__dict__)
                if self._is_same_entity(tracked_e, new_e):
                    # print('same entity')
                    # Update transition matrix
                    # (even if not transitioned, how often the type stays the same is important)
                    self._mark_transition(tracked_e.entity_type, new_e.entity_type)

                    self.tracked_entities[i].update_self(new_e)
                    # print('Updated', self.tracked_entities[i].__dict__)
                    if i in possibly_disappeared:
                        possibly_disappeared.remove(i)
                    del new_entities[new_e_i]
                    break
            else:
                # new entity, and/or tracked_e disappeared
                print('match not found', tracked_e.__dict__)
                print('---')
                possibly_disappeared.append(i)

        for disapp_idx in possibly_disappeared: # well, they definitely disappeared
            # Mark transition in matrix
            self._mark_transition(self.tracked_entities[disapp_idx].entity_type, 'null')

            # Mark object as nonexistent, to be removed in the next timestep
            # We do not delete it in this timestep because we have to show the agent
            # that the entity disappeared
            self.tracked_entities[disapp_idx].disappeared()

        self.do_not_exist.reverse()
        for dne_idx in self.do_not_exist:
            print('DNE', dne_idx)
            del self.tracked_entities[dne_idx]

        self.do_not_exist = newly_nonexistent   # to be removed next time

        for entity_to_add in new_entities:
            entity_to_add.id = self.next_free_entity_id
            entity_to_add.appeared()
            self.tracked_entities.append(entity_to_add)

            # Mark transition in matrix
            self._mark_transition('null', entity_to_add.entity_type)

            # increment id for next appearing object
            self.next_free_entity_id += 1


    def _build_representation(self):
        '''Build time-abstracted representation + object interactions'''

        def interaction(el_1, el_2, loc_diff, types_before, types_after):
            '''Make interaction dict for certain interaction params'''
            return {
                'interaction': (el_1.id, el_2.id),
                'loc_difference': tuple(loc_diff),
                'types_before': types_before,
                'types_after': types_after
            }
        interactions = []
        interactions_built = [] # pairs of entities for which interaction has already been built
        # Build interactions
        for entity in self.tracked_entities:
            within_radius = [x for x in self.tracked_entities
                             if np.all((x.position - entity.position) < self.neighbor_radius*2)]
            for w_r in within_radius:
                sorted_e = (entity, w_r) if entity.entity_type < w_r.entity_type else (w_r, entity)
                interact_ids = (sorted_e[0].id, sorted_e[1].id)
                if interact_ids in interactions_built:
                    #interaction already built before
                    continue

                # position change
                loc_diff = (sorted_e[0].position - sorted_e[0].prev_state['position']) - \
                                    (sorted_e[1].position - sorted_e[1].prev_state['position'])
                types_before = (sorted_e[0].prev_state['entity_type'], sorted_e[1].prev_state['entity_type'])
                types_after = (sorted_e[0].entity_type, sorted_e[1].entity_type)
                if np.array_equal(loc_diff, (0, 0)) and np.array_equal(types_before, types_after):
                    # Empty interaction, don't add it
                    continue
                interactions.append(interaction(sorted_e[0], sorted_e[1], loc_diff,
                                                types_before, types_after))
                interactions_built.append(interact_ids)

        return interactions

    def _mark_transition(self, from_type, to_type):
        '''Mark type transition in transition matrix'''
        self.total_transitions.setdefault(from_type, 0)
        self.total_transitions[from_type] += 1
        self.type_transition_matx.at[from_type, to_type] += 1
