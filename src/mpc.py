import numpy as np


class MPC:
    def __init__(self,
                 env,
                 plan_horizon,
                 model,
                 popsize,
                 num_elites,
                 max_iters,
                 num_particles=6,
                 use_mpc=True,
                 use_gt_dynamics=True,
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

        self.popsize = popsize
        self.max_iters = max_iters
        self.num_elites = num_elites
        self.plan_horizon = plan_horizon
        self.num_particles = num_particles

        self.use_mpc = use_mpc
        self.use_gt_dynamics = use_gt_dynamics
        self.use_random_optimizer = use_random_optimizer

        self.num_nets = None if model is None else model.num_nets

        self.state_dim, self.action_dim = 8, env.action_space.shape[0]
        self.ac_ub, self.ac_lb = env.action_space.high, env.action_space.low

        self.goal = self.env.goal_pos
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
            self.opt = self.random_optimizer
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
        x = max(d_box - 0.4, 0)
        cost = W_PUSHER * x + W_GOAL * d_goal + W_DIFF * diff_coord
        return cost

    def predict_next_state_model(self, states, actions):
        """Given a list of state action pairs, use the learned model to
        predict the next state.

        Returns:
            cost: cost of the given action sequence.
        """
        # Initial cost is the same for all sequences
        rows = actions.shape[0]  # M*P
        cost = np.array([self.obs_cost_fn(states[0, :])] * rows)  # 1200, 5
        sampler = self.ts1sampling(rows)
        for i in range(self.plan_horizon):
            idx = i * self.action_dim
            action = actions[:, idx:idx + self.action_dim]
            idxs = sampler[:, i]
            next_states = self.model.predict(states, action, idxs)
            states = next_states
            cost += np.apply_along_axis(self.obs_cost_fn, axis=1, arr=states)
        return cost

    def predict_next_state_gt(self, states, actions):
        """Given a list of state action pairs, use the ground truth dynamics
        to predict the next state.
        """
        for i in range(self.plan_horizon):
            idx = i * self.action_dim
            action = actions[idx:idx + self.action_dim]
            next_state = self.env.get_nxt_state(states[i], action)
            states.append(next_state)
        return states

    def random_optimizer(self, state):
        """Implements the random optimizer. It gives the best action sequence
        for a certain initial state.
        """
        # Generate M*I action sequences of length T according to N(0, 0.5I)
        total_sequences = self.popsize * self.max_iters
        shape = (total_sequences, self.plan_horizon * self.action_dim)
        self.reset()  # resets mu and sigma
        actions = np.random.normal(self.mu, self.sigma, size=shape)
        actions = np.clip(actions, a_min=-1, a_max=1)
        if not self.use_gt_dynamics:
            repeated_actions = np.tile(actions, reps=(self.num_particles, 1))
            rows = repeated_actions.shape[0]
            states = np.tile(state, reps=(rows, 1))
            costs = self.predict_next_state_model(states, repeated_actions)
            costs = costs.reshape(self.num_particles, -1)
            costs = np.mean(costs, axis=0)  # these are M*I costs
            assert costs.shape[0] == self.popsize * self.max_iters
            min_cost_idx = np.argmin(costs)
            return actions[min_cost_idx]
        else:
            best_cost = np.inf
            best_action_sequence = np.zeros_like(self.mu)
            for i in range(total_sequences):
                states = self.predict_next_state_gt([state], actions[i, :])
                assert len(states) == self.plan_horizon + 1
                cost = sum(self.obs_cost_fn(x) for x in states)
                cost /= self.num_particles
                if cost < best_cost:
                    best_cost = cost
                    best_action_sequence = actions[i, :]
            return best_action_sequence

    def ts1sampling(self, n):
        s = (n, self.plan_horizon)
        return np.random.choice(range(self.num_nets), size=s, replace=True)

    def cem_optimizer(self, state):
        """Implements the Cross Entropy Method optimizer. It gives the action
        sequence for a certain initial state by choosing elite sequences and
        using their mean.
        """
        mu = self.mu
        sigma = self.sigma
        for i in range(self.max_iters):
            # Generate M action sequences of length T according to N(mu, std)
            shape = (self.popsize, self.plan_horizon * self.action_dim)
            actions = np.random.normal(mu, sigma, size=shape)
            actions = np.clip(actions, a_min=-1, a_max=1)
            costs = None
            if not self.use_gt_dynamics:
                reps = (self.num_particles, 1)
                repeated_actions = np.tile(actions, reps=reps)
                rows = repeated_actions.shape[0]
                states = np.tile(state, reps=(rows, 1))
                costs = self.predict_next_state_model(states, repeated_actions)
                costs = costs.reshape(self.num_particles, -1)
                costs = np.mean(costs, axis=0)  # these are M costs
            else:
                costs = list()
                for m in range(self.popsize):
                    states = self.predict_next_state_gt([state], actions[m, :])
                    assert len(states) == self.plan_horizon + 1
                    cost = sum(self.obs_cost_fn(x) for x in states)
                    cost /= self.num_particles
                    costs.append(cost)
            # Calculate mean and std using the elite action sequences
            costs = np.argsort(costs)
            elite_sequences = costs[:self.num_elites]
            elite_actions = actions[elite_sequences, :]
            assert elite_actions.shape[0] == self.num_elites
            mu = np.mean(elite_actions, axis=0)
            sigma = np.std(elite_actions, axis=0)
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
        assert len(obs_trajs) == len(acs_trajs)
        input_states = [traj[:-1, :self.state_dim] for traj in obs_trajs]
        input_states = np.concatenate(input_states, axis=0)
        assert input_states.shape[1] == self.state_dim
        targets = [traj[1:, :self.state_dim] for traj in obs_trajs]
        targets = np.concatenate(targets, axis=0)
        assert targets.shape[1] == self.state_dim
        actions = [acs for acs in acs_trajs]
        actions = np.concatenate(actions, axis=0)
        assert actions.shape[1] == self.action_dim
        inputs = np.concatenate((input_states, actions), axis=1)
        assert inputs.shape[1] == (self.state_dim + self.action_dim)
        self.model.train(inputs, targets, epochs=epochs)

    def reset(self):
        """Initializes variables mu and sigma.
        """
        self.mu = np.zeros(self.plan_horizon * self.action_dim)
        self.reset_sigma()

    def act(self, state, t):
        """
        Find the action for current state.

        Arguments:
          state: current state
          t: current timestep
        """
        self.goal = state[self.state_dim:]
        assert len(self.goal) == 2
        state = state[:self.state_dim]
        if self.use_mpc:
            mu = self.opt(state)
            action = mu[:self.action_dim]  # Get the first action
            action = action.copy()
            mu[:-self.action_dim] = mu[self.action_dim:]
            mu[-self.action_dim:] = 0
            self.mu = mu
        else:
            if t % self.plan_horizon == 0:
                self.mu = self.opt(state)
            idx = (t % self.plan_horizon) * self.action_dim
            action = self.mu[idx:idx + self.action_dim]
        return action

    def reset_sigma(self):
        """Resets/initializes the value of sigma.
        """
        sigma = [0.5 ** 0.5] * (self.plan_horizon * self.action_dim)
        self.sigma = np.array(sigma)
