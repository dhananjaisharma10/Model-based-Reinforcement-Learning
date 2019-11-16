import numpy as np


class Agent:
    def __init__(self, env):
        self.env = env

    def sample(self, horizon, policy):
        """
        Sample a rollout from the agent.

        Arguments:
          horizon (int): the length of the rollout
          policy: the policy that the agent will use for actions
        """
        rewards = []
        states, actions, reward_sum, done = [self.env.reset()], [], 0, False

        policy.reset()
        for t in range(horizon):
            # print('time step: {}/{}'.format(t, horizon))
            actions.append(policy.act(states[t], t))
            state, reward, done, info = self.env.step(actions[t])
            states.append(state)
            reward_sum += reward
            rewards.append(reward)
            if done:
                # print(info['done'])
                break

        # print("Rollout length: ", len(actions))

        return {
            "obs": np.array(states),
            "ac": np.array(actions),
            "reward_sum": reward_sum,
            "rewards": np.array(rewards),
        }
