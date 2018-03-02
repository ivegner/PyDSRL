import argparse

import gym
from gym import logger
from sklearn.model_selection import train_test_split
import numpy as np

import cross_circle_gym
from components import RandomAgent
from components.autoencoder import SymbolAutoencoder
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description=None)
parser.add_argument('env_id', nargs='?', default='CrossCircle-MixedRand-v0', help='Select the environment to run')
parser.add_argument('--load', type=str, help='load existing model from filename provided')
parser.add_argument('--load-train', action='store_true', help='load existing model from filename provided and keep training')
parser.add_argument('--visualize', '--vis', action='store_true', help='plot autoencoder input & output')
parser.add_argument('--save', type=str, help='save model to filename provided')

args = parser.parse_args()

# You can set the level to logger.DEBUG or logger.WARN if you
# want to change the amount of output.
logger.set_level(logger.INFO)
env = gym.make(args.env_id)
seed = env.seed(0)[0]

def make_autoencoder_train_data(num, min_entities=1, max_entities=30):
    temp_env = gym.make('CrossCircle-MixedRand-v0')
    temp_env.seed(0)
    states = []
    for i in range(num):
        states.append(temp_env.make_random_state())
    return np.asarray(states)

logger.info('Making test images...')
images = make_autoencoder_train_data(5000, max_entities=30)
input_shape = images[0].shape + (1,)
if args.load:
    autoencoder = SymbolAutoencoder.from_saved(args.load, input_shape)
else:
    autoencoder = SymbolAutoencoder(input_shape)

logger.info('Splitting sets...')
X_train, X_test = train_test_split(images, test_size=0.2, random_state=seed)
X_train, X_val = train_test_split(X_train, test_size=0.2, random_state=seed)

X_train = np.reshape(X_train, (len(X_train),) + X_train[0].shape + (1,))
X_test = np.reshape(X_test, (len(X_test),) + X_test[0].shape + (1,))
X_val = np.reshape(X_val, (len(X_val),) + X_val[0].shape + (1,))

if args.load_train or not args.load:
    autoencoder.train(X_train, epochs=10, validation=X_val)
if args.save:
    logger.info('Training...')
    autoencoder.save_weights(args.save)

if args.visualize:
    #Visualize autoencoder

    encoded_imgs = autoencoder.encode(X_test)
    encoded_imgs = (encoded_imgs > 1) * encoded_imgs
    encoded_imgs = np.sum(encoded_imgs, axis=3)

    decoded_imgs = autoencoder.predict(X_test)

    n = 10
    plt.figure(figsize=(30, 4))
    for i in range(1, n):
        # display original
        ax = plt.subplot(3, n, i)
        plt.imshow(X_test[i].reshape(images[0].shape))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(3, n, i + n)
        plt.imshow(decoded_imgs[i].reshape(images[0].shape))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display salient features
        ax = plt.subplot(3, n, i + 2*n)
        plt.imshow(encoded_imgs[i].reshape((images[0].shape[0]//2, images[0].shape[1]//2)))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    print('\nPlot visible, close it to proceed')
    plt.show()

# encoded_image = autoencoder.encode(images[0].reshape((1,)+input_shape))
# print(encoded_image.shape)
# print(np.max(encoded_image))

# # agent = RandomAgent(env.action_space)

# steps = 100
# episodes = 10
# reward = 0
# done = False

# ob = env.reset()
# for episode in range(episodes):
#     env.reset()
#     env.render()
#     for step in range(steps):
#         action = agent.act(ob, reward, done)
#         ob, reward, done, info = env.step(action)
#         env.render()
#         print('Action:', action, 'Reward:', reward)
#         if done:
#             break

# env.close()
