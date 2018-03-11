'''Build full state representation'''
class StateRepresentationBuilder():
    '''
    Implements section 3.2 of the paper. Builds state representation \
    of raw entities
    '''
    def __init__(self):
        self.prev_entities = None

    def build_state(self, entities):
        '''Tag entities across time, build interactions'''
        if self.prev_entities is None:
            self.prev_entities = self._init_tracking(entities)
            return

        tracked = self._update_tracking(entities)
        final_repr = self._build_representation(tracked)

        return final_repr


    def _init_tracking(self, entities):
        '''Set up tags for all existing entities'''
        pass

    def _update_tracking(self, new_entities):
        '''Track entities across time, using their last state'''
        pass

    def _build_representation(self, tracked):
        '''Build time-abstracted representation + object interactions'''
        pass