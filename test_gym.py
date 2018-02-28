'''Random-action tester for gym environments'''
import argparse

import gym
from gym import logger
import cross_circle_gym

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='CrossCircle-MixedRand-v0', help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env = gym.make(args.env_id)

    env.seed(0)
    agent = RandomAgent(env.action_space)

    steps = 1000
    episodes = 10
    reward = 0
    done = False

    ob = env.reset()
    for episode in range(episodes):
        env.reset()
        env.render()
        for step in range(steps):
            action = agent.act(ob, reward, done)
            ob, reward, done, info = env.step(action)
            env.render()
            print('Action:', action, 'Reward:', reward)
            if done:
                break

    env.close()
