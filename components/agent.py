'''
MIT License

Copyright (c) 2017 Keon Kim

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

SOURCE: https://github.com/keon/deep-q-learning/blob/master/ddqn.py
'''
import pickle
import random
from collections import deque

import numpy as np
from keras import backend as K
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam


class DDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state,random=True):
        if random:
            if np.random.rand() <= self.epsilon:
                return random.randrange(self.action_size)
            act_values = self.model.predict(state)
        else:
            act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

class TabularAgent:
    '''RL agent as described in the DSRL paper'''
    def __init__(self, action_size,alpha,epsilon_decay,neighbor_radius=25):
        self.action_size = action_size
        self.alpha = alpha
        self.epsilon = 1
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.1
        self.gamma = 0.95
        self.neighbor_radius=neighbor_radius
        self.offset = neighbor_radius*2
        self.tables = {}

    def act(self, state,random_act=True):
        '''
        Determines action to take based on given state
        State: Array of interactions
               (entities in each interaction are presorted by type for consistency)
        Returns: action to take, chosen e-greedily
        '''
        if not random_act:
            return np.argmax(self._total_rewards(state))
        if np.random.rand() <= self.epsilon:
            #print('random action, e:', self.epsilon)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            return random.randrange(self.action_size)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return np.argmax(self._total_rewards(state))  # returns action

    def update(self, state, action, reward, next_state, done):
        '''Update tables based on reward and action taken'''



        for interaction in state:
            type_1, type_2 = interaction['types_after'] # TODO resolve: should this too be types_before?
            table = self.tables.setdefault(type_1, {}).setdefault(type_2, self._make_table())
            id1,id2 = interaction['interaction']
            interaction_next_state = [inter for inter in next_state if inter['interaction']==(id1,id2)]
            if len(interaction_next_state)==0:
                continue
            elif len(interaction_next_state)>1:
                raise ValueError('This should not happen')
            else:
                #print('Now we should update the Q-values')
                #print(f'The current reward is {reward}')
                interaction_next_state = interaction_next_state[0]
                interaction['loc_difference'] = (interaction['loc_difference'][0]+self.offset,interaction['loc_difference'][1]+self.offset)
                interaction_next_state['loc_difference'] = (interaction_next_state['loc_difference'][0]+self.offset,interaction_next_state['loc_difference'][1]+self.offset)
                #print(interaction_next_state['loc_difference'])
                #print(interaction['loc_difference'])
                next_action_value = table[interaction_next_state['loc_difference']]
                #print(f'The next action value {next_action_value}')
                if done:
                    table[interaction['loc_difference']][action] = reward
                else:
                    #print(f'Q-value before update {table[interaction["loc_difference"]][action]}')
                    #print(f'Location {interaction["loc_difference"]}')
                    #print(f"The new value should be {table[interaction['loc_difference']][action] + self.alpha*(reward + self.gamma * np.max(next_action_value) - table[interaction['loc_difference']][action])}")
                    #print(interaction['loc_difference'])
                    table[interaction['loc_difference']][action] = table[interaction['loc_difference']][action] + self.alpha*(reward + self.gamma * np.max(next_action_value) - table[interaction['loc_difference']][action])
                    #print(f'Q-value after update {table[interaction["loc_difference"]][action]}')

    def _total_rewards(self, interactions):
        action_rewards = np.zeros(self.action_size)
        for interaction in interactions:
            type_1, type_2 = interaction['types_before']
            table = self.tables.setdefault(type_1, {}).setdefault(type_2, self._make_table())
            action_rewards += table[interaction['loc_difference']]  # add q-value arrays
        return action_rewards

    def _make_table(self):
        '''
        Makes table for q-learning
        3-D table: rows = loc_difference_x, cols = loc_difference_y, z = q-values for actions
        Rows and cols added to as needed
        '''
        return np.zeros((self.neighbor_radius * 8, self.neighbor_radius * 8, self.action_size),
                        dtype=float)

    def save(self, filename):
        '''Save agent's tables'''
        with open(filename, 'wb') as f_p:
            pickle.dump(self.tables, f_p)

    @staticmethod
    def from_saved(filename, action_size):
        '''Load agent from filename'''
        with open(filename, 'rb') as f_p:
            tables = pickle.load(f_p)
            ret = TabularAgent(action_size)
            assert len(tables.values()[0].values()[0][0, 0]) == action_size, \
                   'Action size given to from_saved doesn\'t match the one in the tables'
            ret.tables = tables
        return ret
