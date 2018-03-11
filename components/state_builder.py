'''Build full state representation'''
import numpy as np

class StateRepresentationBuilder():
    '''
    Implements section 3.2 of the paper. Builds state representation \
    of raw entities
    '''
    def __init__(self):
        self.tracked_entities = []
        self.next_free_entity_id = 0
        self.type_transition_matx =

    def build_state(self, entities):
        '''Tag entities across time, build interactions'''
        if not self.tracked_entities:
            self._init_tracking(entities)
        else:
            self._update_tracking(entities)

        final_repr = self._build_representation(self.tracked_entities)

        return final_repr

    def _init_tracking(self, entities):
        '''Set up tags for all existing entities'''
        for e_type, e_array in new_entities.items():
            e = Entity(e, e_type, tracking_id=self.next_free_entity_id)
            self.tracked_entities.append(e)
            self.next_free_entity_id += 1

    def _is_same_entity(self, e1, e2):
        '''Check whether e1 is a displaced version of e2'''
        # similarity = 0

        # l_dist = 1 / (1 + np.linalg.norm(e1.position - e2.position))
        # TODO
        pass

    def _update_tracking(self, new_entities):
        '''Track entities across time, using their last state'''
        new_entity_objs = []
        for e_type, e_array in new_entities.items():
            new_entity_objs.extend(map(lambda e: Entity(e, e_type), e_array))
        # if an entity is not matched with any in new_entities,
        # place it in possibly_disappeared, and remove it if encountered
        # If there are any in possibly_disappeared by the time the
        # iteration over new_entities is done, the entity has actually disappeared
        possibly_disappeared = []
        do_not_exist = []
        for i, tracked_e in enumerate(self.tracked_entities):
            if not tracked_e.exists:
                do_not_exist.append(i)
                continue
            for new_e_i, new_e in enumerate(new_entity_objs):
                if self._is_same_entity(tracked_e, new_e):
                    self.tracked_entities[i].update_self(new_e)
                    if i in possibly_disappeared:
                        possibly_disappeared.remove(i)
                    del new_entity_objs[new_e_i]
                    break
            else:
                # new entity, and/or tracked_e disappeared
                possibly_disappeared.append(i)

        for disapp_idx in possibly_disappeared: # well, they definitely disappeared
            self.tracked_entities[disapp_idx].disappeared()

        do_not_exist.reverse()
        for dne_idx in do_not_exist:
            del self.tracked_entities[dne_idx]

        for entity_to_add in new_entity_objs:
            entity_to_add.id = self.next_free_entity_id
            entity_to_add.appeared()
            self.tracked_entities.append(entity_to_add)

            self.next_free_entity_id += 1

    def _build_representation(self, tracked):
        '''Build time-abstracted representation + object interactions'''
        # TODO
        pass

class Entity():
    '''Class for an entity that has an id, attributes, etc.'''
    def __init__(self, position, type, tracking_id=None):
        self.position = position
        self.type = type
        self.id = tracking_id
        self.last_transition = None
        self.exists = 1

    def update_self(self, entity):
        '''Update own params based on new entity object'''
        self.position = entity.position
        if entity.type != self.type:
            # transitioned to new type
            self._transition(self.type, entity.type)
        else:
            # unset old transition if any
            self.last_transition = None


    def _transition(self, from_type, to_type):
        '''Set transition for object'''
        self.last_transition = [from_type, to_type]
        self.type = to_type

    def appeared(self):
        '''Set object transition as having spawned'''
        self._transition('null', self.type)

    def disappeared(self):
        '''Set object transition as despawned, and mark it for deletion'''
        self._transition(self.type, 'null')
        self.exists = 0 # mark for deletion