import os
import gym
import envs
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from collections import deque
from mpc import MPC
from model import PENN
from agent import Agent, RandomPolicy

# Training params
TASK_HORIZON = 40
PLAN_HORIZON = 5

# CEM params
POPSIZE = 200
NUM_ELITES = 20
MAX_ITERS = 5

# Model params
LR = 1e-3

# Dims
STATE_DIM = 8

LOG_DIR = './data'


class ExperimentGTDynamics(object):
    def __init__(self, env_name='Pushing2D-v1', mpc_params=None):
        self.env = gym.make(env_name)
        self.task_horizon = TASK_HORIZON

        self.agent = Agent(self.env)
        # Does not need model
        self.warmup = False
        mpc_params['use_gt_dynamics'] = True
        self.cem_policy = MPC(self.env, PLAN_HORIZON, None, POPSIZE,
                              NUM_ELITES, MAX_ITERS, **mpc_params,
                              use_random_optimizer=False)
        self.random_policy = MPC(self.env, PLAN_HORIZON, None, POPSIZE,
                                 NUM_ELITES, MAX_ITERS, **mpc_params,
                                 use_random_optimizer=True)

    def test(self, num_episodes, optimizer='cem'):
        samples = []
        print('Optimizer:', optimizer)
        for j in range(num_episodes):
            print('Test episode {}'.format(j))
            print('')
            samples.append(
                self.agent.sample(
                    self.task_horizon, self.cem_policy if
                    optimizer == 'cem' else self.random_policy
                )
            )
        avg_return = np.mean([sample["reward_sum"] for sample in samples])
        avg_success = np.mean([sample["rewards"][-1] == 0
                               for sample in samples])
        return avg_return, avg_success


class ExperimentModelDynamics:
    def __init__(self, env_name='Pushing2D-v1', num_nets=1, mpc_params=None):
        self.env = gym.make(env_name)
        self.task_horizon = TASK_HORIZON

        self.agent = Agent(self.env)
        mpc_params['use_gt_dynamics'] = False
        self.model = PENN(num_nets, STATE_DIM, len(self.env.action_space.sample()), LR)
        self.cem_policy = MPC(self.env, PLAN_HORIZON, self.model, POPSIZE, NUM_ELITES, MAX_ITERS, **mpc_params,
                              use_random_optimizer=False)
        self.random_policy = MPC(self.env, PLAN_HORIZON, self.model, POPSIZE, NUM_ELITES, MAX_ITERS, **mpc_params,
                                 use_random_optimizer=True)
        self.random_policy_no_mpc = RandomPolicy(len(self.env.action_space.sample()))

    def test(self, num_episodes, optimizer='cem'):
        samples = []
        for j in range(num_episodes):
            print('Test episode {}'.format(j))
            samples.append(
                self.agent.sample(
                    self.task_horizon, self.cem_policy
                    if optimizer == 'cem' else self.random_policy
                )
            )
        avg_return = np.mean([sample["reward_sum"] for sample in samples])
        avg_success = np.mean([sample["rewards"][-1] == 0 for sample in samples])
        return avg_return, avg_success

    def model_warmup(self, num_episodes, num_epochs):
        """Train a single probabilistic model using a random policy
        """
        # samples = list()
        # for i in range(num_episodes):
        #     samples.append(self.agent.sample(
        #         self.task_horizon, self.random_policy_no_mpc))
        samples = np.load('samples.npy', allow_pickle=True)
        print('')
        print('*'*20, 'TRAINING', '*'*20)
        self.cem_policy.train(
            [sample["obs"] for sample in samples],
            [sample["ac"] for sample in samples],
            [sample["rewards"] for sample in samples],
            epochs=num_epochs
        )

    def train(self,
              num_train_epochs,
              num_episodes_per_epoch,
              evaluation_interval,
              use_buffer=True):
        """Jointly training the model and the policy
        """

        def save(filename, arr):
            np.save(filename, arr)

        # Buffer has a max size of 100 episodes worth of data
        obs_buffer = deque(maxlen=40 * 100)
        ac_buffer = deque(maxlen=40 * 100)
        rewards_buffer = deque(maxlen=40 * 100)
        success_cem = list()
        success_random = list()
        success_name_cem = 'success_pets_cem.npy'
        success_name_ran = 'success_pets_random.npy'
        for i in range(num_train_epochs):
            print("Starting training epoch %d." % (i + 1))
            # Collect data using the current dynamics model
            samples = list()
            for j in range(num_episodes_per_epoch):
                samples.append(
                    self.agent.sample(self.task_horizon, self.cem_policy))
            # Update replay buffers
            new_obs = [sample["obs"] for sample in samples]
            new_ac = [sample["ac"] for sample in samples]
            new_rewards = [sample["rewards"] for sample in samples]
            obs_buffer.extend(new_obs)
            ac_buffer.extend(new_ac)
            rewards_buffer.extend(new_rewards)
            print("Rewards obtained:", [sample["reward_sum"]
                  for sample in samples])
            # Train the dynamics model using the data
            if use_buffer:
                self.cem_policy.train(obs_buffer, ac_buffer, rewards_buffer,
                                      epochs=5)
            else:
                self.cem_policy.train(new_obs, new_ac, new_rewards, epochs=5)
            # Test
            num_test_episodes = 20
            if (i + 1) % evaluation_interval == 0:
                avg_return, avg_success_cem = self.test(
                    num_test_episodes, optimizer='cem')
                print('Test success CEM + MPC:', avg_success_cem)
                success_cem.append(avg_success_cem)
                save(success_name_cem, success_cem)
                avg_return, avg_success_ran = self.test(
                    num_test_episodes, optimizer='random')
                print('Test success Random + MPC:', avg_success_ran)
                success_random.append(avg_success_ran)
                save(success_name_ran, success_random)


def test_cem_gt_dynamics(num_episode=10):
    # mpc_params = {'use_mpc': False, 'num_particles': 1}
    # exp = ExperimentGTDynamics(env_name='Pushing2D-v1', mpc_params=mpc_params)
    # avg_reward, avg_success = exp.test(num_episode)
    # print('CEM PushingEnv: avg_reward: {}, avg_success: {}'.format(avg_reward, avg_success))

    # mpc_params = {'use_mpc': True, 'num_particles': 1}
    # exp = ExperimentGTDynamics(env_name='Pushing2D-v1', mpc_params=mpc_params)
    # avg_reward, avg_success = exp.test(num_episode)
    # print('MPC PushingEnv: avg_reward: {}, avg_success: {}'.format(avg_reward, avg_success))

    # mpc_params = {'use_mpc': False, 'num_particles': 1}
    # exp = ExperimentGTDynamics(env_name='Pushing2DNoisyControl-v1', mpc_params=mpc_params)
    # avg_reward, avg_success = exp.test(num_episode)
    # print('CEM PushingEnv Noisy: avg_reward: {}, avg_success: {}'.format(avg_reward, avg_success))

    # mpc_params = {'use_mpc': True, 'num_particles': 1}
    # exp = ExperimentGTDynamics(env_name='Pushing2DNoisyControl-v1', mpc_params=mpc_params)
    # avg_reward, avg_success = exp.test(num_episode)
    # print('MPC PushingEnv Noisy: avg_reward: {}, avg_success: {}'.format(avg_reward, avg_success))

    mpc_params = {'use_mpc': False, 'num_particles': 1}
    exp = ExperimentGTDynamics(env_name='Pushing2D-v1', mpc_params=mpc_params)
    avg_reward, avg_success = exp.test(num_episode, optimizer='random')
    print('MPC PushingEnv Noisy: avg_reward: {}, avg_success: {}'.format(avg_reward, avg_success))

    mpc_params = {'use_mpc': True, 'num_particles': 1}
    exp = ExperimentGTDynamics(env_name='Pushing2D-v1', mpc_params=mpc_params)
    avg_reward, avg_success = exp.test(num_episode, optimizer='random')
    print('MPC PushingEnv Noisy: avg_reward: {}, avg_success: {}'.format(avg_reward, avg_success))


def train_single_dynamics(num_test_episode=50):
    num_nets = 1
    num_episodes = 1000
    num_epochs = 100
    mpc_params = {'use_mpc': True, 'num_particles': 1}
    exp = ExperimentModelDynamics(env_name='Pushing2D-v1', num_nets=num_nets,
                                  mpc_params=mpc_params)
    print('*'*20, 'WARMUP', '*'*20)
    print('')
    exp.model_warmup(num_episodes=num_episodes, num_epochs=num_epochs)
    print('*'*20, 'TEST', '*'*20)
    avg_reward, avg_success = exp.test(num_test_episode, optimizer='random')
    print('MPC PushingEnv: avg_reward: {}, avg_success: {}'.format(
        avg_reward, avg_success))


def train_pets():
    num_nets = 2
    num_epochs = 500
    evaluation_interval = 50
    num_episodes_per_epoch = 1

    mpc_params = {'use_mpc': True, 'num_particles': 6}
    exp = ExperimentModelDynamics(env_name='Pushing2D-v1', num_nets=num_nets, mpc_params=mpc_params)
    print('*'*20, 'WARMUP', '*'*20)
    exp.model_warmup(num_episodes=100, num_epochs=10)
    exp.train(num_train_epochs=num_epochs,
              num_episodes_per_epoch=num_episodes_per_epoch,
              evaluation_interval=evaluation_interval)


if __name__ == "__main__":
    # Disable AVX/FMA and TF warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    # test_cem_gt_dynamics(50)
    # train_single_dynamics(50)
    train_pets()
