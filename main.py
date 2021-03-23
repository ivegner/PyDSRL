'''Main module for the paper's algorithm'''

import argparse
import os

from collections import deque
from datetime import datetime


import numpy as np
import tensorflow as tf
import tqdm

from gym import logger

import cross_circle_gym

from components.state_builder import StateRepresentationBuilder
from components.agent import TabularAgent
from utils import prepare_training

# Experiment Parameters
parser = argparse.ArgumentParser(description=None)
parser.add_argument('--experiment_name', type=str, default='default', help='Name of the experiment')
parser.add_argument('--load', type=str, help='load existing model from filename provided')
parser.add_argument('--image_dir', type=str, help='laod images from directory provided')
parser.add_argument('--episodes', '-e', type=int, default=1000,
                    help='number of DQN training episodes')
parser.add_argument('--load-train', action='store_true',
                    help='load existing model from filename provided and keep training')
parser.add_argument('--new-images', action='store_true', help='make new set of training images')
parser.add_argument('--enhancements', action='store_true',
                    help='activate own improvements over original paper')
parser.add_argument('--visualize', '--vis', action='store_true',
                    help='plot autoencoder input & output')
parser.add_argument('--save', type=str, help='save model to directory provided')
parser.add_argument('--logdir',type=str,default='./logs', help='Log directory')
parser.add_argument('--log_level',type=str,default='warn',help='Detail of logging output')
parser.add_argument('--evaluation_frequency', type=int, default=100,
                    help='How often to evaluate the agent')
parser.add_argument('--tensorboard', action='store_true', default=False,
                    help='Switch on tensorboard for the autoencoder training')
parser.add_argument('--play', action='store_true', default=False,
                    help='Choose the agents action for 20 timesteps to see what the autoencoder does')

# Environment
parser.add_argument('--random', action='store_true', default=False,
                    help='Should the position of the entities be random')
parser.add_argument('--double', action='store_true', default=False,
                    help='Only negative objects (circles) or also positive ones (cross)')
parser.add_argument('--n_entities', type=int, default=16,
                    help='Number of entities in the environment')
parser.add_argument('--entity_size', type=int, default=10, help='Size of the entities')
parser.add_argument('--neighborhood_size', type=int, default=10,
                    help='Size of the neighborhood')
parser.add_argument('--step_size', type=float, default=1.0, help='Step-Size')
parser.add_argument('--overlap_factor', type=float, default=0.01,
                    help='How much must an gent overlap with an entitiy to collect it')
parser.add_argument('--colour_state', action='store_true', default=False,
                    help='Whether to use the colour image as a state or a one-channel black and white image')

# Training parameters
parser.add_argument('--alpha', type=float, default=0.01, help='Learning Rate')
parser.add_argument('--epsilon_decay', type=float, default=0.99995,
                    help='Decay rate of epsilon')
parser.add_argument('--timesteps', type=int, default=100, help='Length of a training episode')

# Autoencdoer
parser.add_argument('--filter_size', default=10, type=int, help='Size of the filter')


args = parser.parse_args()

now = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
args.logdir = os.path.join(args.logdir,args.experiment_name,now)

# Choose environment
if args.random and args.double:
    env_id = 'CrossCircle-MixedRand-v0'
elif args.random and not args.double:
    env_id = 'CrossCircle-NegRand-v0'
elif not args.random and args.double:
    env_id = 'CrossCircle-MixedGrid-v0'
else:
    env_id = 'CrossCircle-NegGrid-v0'
args.env_id = env_id

# Set logger
if args.log_level=='warn':
    logger.setLevel(logger.WARN)
elif args.log_level=='info':
    logger.setLevel(logger.INFO)
else:
    raise NotImplementedError('Log-level not implemented')
args.logger = logger

autoencoder,env = prepare_training(args)

state_builder = StateRepresentationBuilder(neighbor_radius=args.neighborhood_size)
action_size = env.action_space.n
agent = TabularAgent(action_size,args.alpha,args.epsilon_decay,args.neighborhood_size)

done = False
time_steps = args.timesteps

number_of_evaluations = 0
buffered_rewards = deque(maxlen=200)

summary_writer = tf.summary.create_file_writer(args.logdir)

for e in tqdm.tqdm(range(args.episodes)):
    state_builder.restart()
    state = env.reset()
    state = state_builder.build_state(*autoencoder.get_entities(state))
    total_reward = 0

    for t in range(time_steps):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        next_state = state_builder.build_state(*autoencoder.get_entities(next_state))
        agent.update(state, action, reward, next_state, done)
        state = next_state
        if done:
            break

    buffered_rewards.append(total_reward)

    with summary_writer.as_default():
        tf.summary.scalar('Averaged Reward',np.mean(buffered_rewards),e)
        tf.summary.scalar('Epsilon',agent.epsilon,e)


    if e % args.evaluation_frequency == 0:
        number_of_evaluations += 1
        agent.save(os.path.join(args.logdir,'tab_agent.h5'))
        evaluation_reward = []
        with summary_writer.as_default():
            for i in range(10):
                done = False
                state_builder.restart()
                image = env.reset()
                state = state_builder.build_state(*autoencoder.get_entities(image))
                total_reward = 0
                for t in range(time_steps):
                    action = agent.act(state,random_act=False)
                    next_image, reward, done, _ = env.step(action)
                    if i==0:
                        tf.summary.image(f'Agent Behaviour {number_of_evaluations}',np.reshape(image,(1,)+image.shape),t)
                    total_reward += reward
                    next_state = state_builder.build_state(*autoencoder.get_entities(next_image))
                    state = next_state
                    image = next_image
                evaluation_reward.append(total_reward)

            tf.summary.scalar('Evaluation Reward',np.mean(evaluation_reward),number_of_evaluations)







