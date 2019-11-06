import os
import tensorflow as tf
import numpy as np
import gym
import copy


class MPC:
    def __init__(self,
                 env,
                 plan_horizon,
                 model,
                 popsize,
                 num_elites,
                 max_iters,
                 num_particles=6,
                 use_gt_dynamics=True,
                 use_mpc=True,
                 use_random_optimizer=False):
        """
        :param env:
        :param plan_horizon:
        :param model: The learned dynamics model to use, which can be None if
            use_gt_dynamics is True
        :param popsize: Population size
        :param num_elites: CEM parameter
        :param max_iters: CEM parameter
        :param num_particles: Number of trajectories for TS1
        :param use_gt_dynamics: Whether to use the ground truth dynamics from
            the environment
        :param use_mpc: Whether to use only the first action of a planned
            trajectory
        :param use_random_optimizer: Whether to use CEM or take random actions
        """

        self.env = env
        self.use_gt_dynamics = use_gt_dynamics
        self.use_mpc = use_mpc
        self.use_random_optimizer = use_random_optimizer
        self.plan_horizon = plan_horizon
        self.popsize = popsize
        self.num_elites = num_elites
        self.max_iters = max_iters
        self.num_particles = num_particles
        self.num_nets = None if model is None else model.num_nets

        self.state_dim, self.action_dim = 8, env.action_space.shape[0]
        self.ac_ub, self.ac_lb = env.action_space.high, env.action_space.low

        # Set up optimizer
        self.model = model

        if use_gt_dynamics:
            self.predict_next_state = self.predict_next_state_gt
            assert num_particles == 1
        else:
            self.predict_next_state = self.predict_next_state_model

        # Initialize your planner with the relevant arguments.
        # Write different optimizers for cem and random actions respectively
        if self.use_random_optimizer:
            raise NotImplementedError
        else:
            self.opt = self.cem_optimizer

    def obs_cost_fn(self, state):
        """ Cost function of the current state """
        # Weights for different terms
        W_PUSHER = 1
        W_GOAL = 2
        W_DIFF = 5

        pusher_x, pusher_y = state[0], state[1]
        box_x, box_y = state[2], state[3]
        goal_x, goal_y = self.goal[0], self.goal[1]

        pusher_box = np.array([box_x - pusher_x, box_y - pusher_y])
        box_goal = np.array([goal_x - box_x, goal_y - box_y])
        d_box = np.sqrt(np.dot(pusher_box, pusher_box))
        d_goal = np.sqrt(np.dot(box_goal, box_goal))
        diff_coord = np.abs(box_x / box_y - goal_x / goal_y)
        # the -0.4 is to adjust for the radius of the box and pusher
        return W_PUSHER * np.max(d_box - 0.4, 0) + W_GOAL * d_goal + W_DIFF * diff_coord

    def predict_next_state_model(self, states, actions):
        """ Given a list of state action pairs, use the learned model to
        predict the next state
        """
        # TODO: write your code here
        raise NotImplementedError

    def predict_next_state_gt(self, states, actions):
        """ Given a list of state action pairs, use the ground truth dynamics
        to predict the next state

        Returns:
            n_s_t (list): States predicted from the ground truth
            dynamics. (TODO: Verify whether this should be a list)
        """
        n_s_t = [self.env.get_nxt_state(s, a) for s, a in zip(states, actions)]
        return n_s_t

    def cem_optimizer(self, mu, sigma):
        # TODO: Generate M action sequences of length T according to
        # N(mu, sigma). Verify your method.
        actions = np.random.normal(mu, sigma, size=(self.popsize,
                                                    self.plan_horizon))
        for i in range(self.max_iters):
            q = {}
            for m in range(self.popsize):
                cost = 0
                # TODO: yet to implement for learned model
                if not self.use_gt_dynamics:
                    raise NotImplementedError
                new_state = self.env.reset()
                cost += self.obs_cost_fn(new_state)
                for a in range(self.plan_horizon):
                    state = new_state
                    action = actions[m, a]
                    new_state, _, _, _ = self.predict_next_state(state, action)
                    cost += self.obs_cost_fn(new_state)
                q[cost] = m
            # Caclulate mean and std using the elite action sequences
            q = sorted(q.items(), key=lambda x: x[0])
            q = q[:self.num_elites]
            elite_actions = actions[[x[1] for x in q]]
            # TODO: Verify your method.
            mu = np.mean(elite_actions)
            sigma = np.std(elite_actions)

        return mu

    def train(self, obs_trajs, acs_trajs, rews_trajs, epochs=5):
        """
        Take the input obs, acs, rews and append to existing transitions the
        train model.

        Args:
          obs_trajs: states
          acs_trajs: actions
          rews_trajs: rewards (NOTE: this may not be used)
          epochs: number of epochs to train for
        """
        # TODO: write your code here
        raise NotImplementedError

    def reset(self):
        # TODO: write your code here
        raise NotImplementedError

    def act(self, state, t):
        """
        Use model predictive control to find the action give current state.

        Arguments:
          state: current state
          t: current timestep
        """
        # TODO: write your code here
        raise NotImplementedError

    # TODO: write any helper functions that you need
