# Some of the code from the following sources:
# https://github.com/joschu/modular_rl
# https://github.com/Khrylx/PyTorch-RL
# http://www.johndcook.com/blog/standard_deviation

import numpy as np


def estimate_advantages(rewards, masks, values, gamma, tau):
    deltas = np.zeros(len(rewards))
    advantages = np.zeros(len(rewards))

    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(len(rewards))):
        deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values[i]
        advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]

        prev_value = values[i]
        prev_advantage = advantages[i]

    returns = values + advantages
    advantages = (advantages - np.mean(advantages)) / np.std(advantages)

    return advantages, returns


class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape


class ZFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip
        self.shape = shape

        self.rs = RunningStat(shape)

    def reset(self):
        self.rs = RunningStat(self.shape)

    def fit(self, X):
        for x in X:
            self.rs.push(x)

    def __call__(self, x, update=True):
        if update:
            self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x
