'''Random-action tester for gym environments'''
import argparse
import pprint
import gym
from gym_recording.wrappers import TraceRecordingWrapper
import os

import cross_circle_gym  # Required, registers the environments.


class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


class Filter(object):
    def __init__(self, m):
        self.m = m

    def __call__(self, n):
        return n % self.m == 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('directory')
    parser.add_argument('--n-steps', type=int, default=100)
    parser.add_argument('--n-episodes', type=int, default=100)
    parser.add_argument('env_id', nargs='?', default='CrossCircle-MixedRand-v0')
    args = parser.parse_args()

    os.makedirs(args.directory, exist_ok=True)
    with open(os.path.join(args.directory, 'config.txt'), 'w') as f:
        f.write(pprint.pformat(args))

    env = gym.make(args.env_id)

    os.makedirs(args.directory, exist_ok=True)

    env = TraceRecordingWrapper(
        env, directory=args.directory, episode_filter=Filter(1), frame_filter=Filter(1))

    env.seed(0)
    agent = RandomAgent(env.action_space)

    reward = 0
    done = False

    ob = env.reset()
    for episode in range(args.n_episodes):
        env.reset()
        # env.render()
        for step in range(args.n_steps):
            action = agent.act(ob, reward, done)
            ob, reward, done, info = env.step(action)
            # env.render()
            print('Action:', action, 'Reward:', reward)
            if done:
                break

    env.close()
