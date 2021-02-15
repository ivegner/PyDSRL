'''Main module for the paper's algorithm'''
#pylint:disable=C0103,R0913
import argparse
import os.path
import pickle
import numpy as np

import gym
from gym import logger
from sklearn.model_selection import train_test_split

#pylint:disable=W0611
import cross_circle_gym
#pylint:enable=W0611
from components.autoencoder import SymbolAutoencoder
from components.state_builder import StateRepresentationBuilder
from components.agent import TabularAgent #, DDQNAgent

parser = argparse.ArgumentParser(description=None)
parser.add_argument('env_id', nargs='?', default='CrossCircle-MixedRand-v0',
                    help='Select the environment to run')
parser.add_argument('--load', type=str, help='load existing model from filename provided')
parser.add_argument('--episodes', '-e', type=int, default=1000,
                    help='number of DQN training episodes')
parser.add_argument('--load-train', action='store_true',
                    help='load existing model from filename provided and keep training')
parser.add_argument('--new-images', action='store_true', help='make new set of training images')
parser.add_argument('--enhancements', action='store_true',
                    help='activate own improvements over original paper')
parser.add_argument('--visualize', '--vis', action='store_true',
                    help='plot autoencoder input & output')
parser.add_argument('--save', type=str, help='save model to filename provided')

args = parser.parse_args()

TRAIN_IMAGES_FILE = 'train_images.pkl'
NEIGHBOR_RADIUS = 25  # 1/2 side of square in which to search for neighbors

# You can set the level to logger.DEBUG or logger.WARN if you
# want to change the amount of output.
logger.setLevel(logger.INFO)



env = gym.make(args.env_id)
seed = env.seed(1)[0]



def make_autoencoder_train_data(num, min_entities=1, max_entities=30):
    '''Make training images for the autoencoder'''
    temp_env = gym.make('CrossCircle-MixedRand-v0')
    temp_env.seed(0)
    states = []
    for _ in range(num):
        states.append(temp_env.make_random_state(min_entities, max_entities))
    return np.asarray(states)

if not os.path.exists(TRAIN_IMAGES_FILE) or args.new_images:
    logger.info('Making test images...')
    images = make_autoencoder_train_data(5000, max_entities=30)
    with open(TRAIN_IMAGES_FILE, 'wb') as f:
        pickle.dump(images, f)
else:
    logger.info('Loading test images...')
    with open(TRAIN_IMAGES_FILE, 'rb') as f:
        images = pickle.load(f)

#input_shape = images[0].shape + (1,)
input_shape = images[0].shape
if args.load:
    autoencoder = SymbolAutoencoder.from_saved(args.load,
                                               images[0].shape,
                                               neighbor_radius=NEIGHBOR_RADIUS)
else:
    autoencoder = SymbolAutoencoder(images[0].shape, neighbor_radius=NEIGHBOR_RADIUS)

if args.load_train or args.visualize or not args.load:
    logger.info('Splitting sets...')
    X_train, X_test = train_test_split(images, test_size=0.2, random_state=seed)
    X_train, X_val = train_test_split(X_train, test_size=0.2, random_state=seed)

    if args.load_train or not args.load:
        logger.info('Training...')
        autoencoder.train(X_train, epochs=10, validation=X_val)

    if args.visualize:
        #Visualize autoencoder
        vis_imgs = X_test[:10]
        autoencoder.visualize(vis_imgs)

if args.save:
    autoencoder.save_weights(args.save)


# entities, found_types = autoencoder.get_entities(X_test[0])

state_builder = StateRepresentationBuilder()
# state = state_builder.build_state(entities, found_types)
# print(state)

# state_size = None # TODO
action_size = env.action_space.n
# if args.enhancements:
#     agent = DDQNAgent(state_size, action_size)
# else:
agent = TabularAgent(action_size)
# # agent.load('./save/cartpole-ddqn.h5')
done = False
batch_size = 32
time_steps = 100

for e in range(args.episodes):
    state_builder.restart()
    state = env.reset()
    state = np.reshape(state, input_shape)
    state = state_builder.build_state(*autoencoder.get_entities(state))
    for time in range(time_steps):
        env.render(wait=1)
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, input_shape)
        next_state = state_builder.build_state(*autoencoder.get_entities(next_state))
        # next_state = np.reshape(next_state, [1, state_size])
        agent.update(state, action, reward, next_state, done)
        state = next_state
        if done:
            break
    # if args.enhancements:
    #     agent.update_target_model()
    print('episode: {}/{}, e: {:.2}'
          .format(e, args.episodes, agent.epsilon))

    # if len(agent.memory) > batch_size:
    #     agent.replay(batch_size)
    if e % 10 == 0:
        agent.save('tab_agent.h5')
