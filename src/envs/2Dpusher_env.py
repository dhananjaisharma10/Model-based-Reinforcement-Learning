import Box2D
from Box2D.b2 import (circleShape, fixtureDef, polygonShape)
from opencv_draw import OpencvDrawFuncs
import cv2

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import copy

MIN_COORD = 0
MAX_COORD = 5
PUSHER_START = np.array([1.0, 1.0])
BOX_START = np.array([2.0, 2.0])
FORCE_MULT = 1
RAD = 0.2
SIDE_GAP_MULT = 2
BOX_RAD = 0.2
GOAL_RAD = 0.5
MAX_STEPS = 40
FPS = 4


class Pusher2d(gym.Env):
    def __init__(self, control_noise=0.):
        self.control_noise = control_noise
        self.seed()
        self.world = Box2D.b2World(gravity=(0, 0))
        self.pusher = None
        self.box = None
        # Actions: x-movement, y-movement (clipped -1 to 1)
        self.action_space = spaces.Box(np.ones(2) * -1, np.ones(2), dtype=np.float32)
        # State: pusher xy position, box xy position, pusher xy velocity, box xy velocity, goal xy position
        self.observation_space = spaces.Box(np.ones(10) * MIN_COORD, np.ones(10) * MAX_COORD, dtype=np.float32)
        self.reset()
        self.drawer = OpencvDrawFuncs(w=240, h=180, ppm=40)
        self.drawer.install()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def random_place(self):
        """ returns [x, y] within an area slightly away from the initial box position """
        return [self.np_random.uniform(BOX_START[0] + BOX_RAD + GOAL_RAD, MAX_COORD - RAD * SIDE_GAP_MULT),
                self.np_random.uniform(BOX_START[1] + BOX_RAD + GOAL_RAD, MAX_COORD - RAD * SIDE_GAP_MULT)]

    def _destroy(self):
        """ removes instantiated Box2D entities """
        if not self.box:
            return
        self.world.DestroyBody(self.box)
        self.world.DestroyBody(self.pusher)

    def reset(self):
        """ standard Gym method; returns first state of episode """
        self._destroy()
        self.pusher = self.world.CreateDynamicBody(
            position=PUSHER_START[:],
            fixtures=fixtureDef(
                shape=circleShape(radius=RAD, pos=(0, 0)),
                density=1.0
            )
        )
        self.box = self.world.CreateDynamicBody(
            position=BOX_START[:],
            fixtures=fixtureDef(
                shape=circleShape(radius=BOX_RAD, pos=(0, 0)),
                density=1.0
            )
        )
        self.goal_pos = self.random_place()
        self.elapsed_steps = 0
        return self._get_obs()

    def step(self, action, render=False):
        """ standard Gym method; returns s, r, d, i """
        if render:
            self.drawer.clear_screen()
            self.drawer.draw_world(self.world)

        action = np.clip(action, -1, 1).astype(np.float32)
        if self.control_noise > 0.:
            action += np.random.normal(0., scale=self.control_noise, size=action.shape)

        self.elapsed_steps += 1
        self.pusher._b2Body__SetLinearVelocity((FORCE_MULT * action[0], FORCE_MULT * action[1]))
        self.box._b2Body__SetActive(True)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        if render:
            cv2.imshow("world", self.drawer.screen)
            cv2.waitKey(20)
        done = False
        reward = -1
        obj_coords = np.concatenate([self.pusher.position.tuple, self.box.position.tuple])
        info = {"done": None}
        # check if out of bounds
        if np.min(obj_coords) < MIN_COORD or np.max(obj_coords) > MAX_COORD:
            reward = -1 * (MAX_STEPS - self.elapsed_steps + 2)
            done = True
            info['done'] = 'unstable simulation'
        # check if out of time
        elif self.elapsed_steps >= MAX_STEPS:
            done = True
            info["done"] = "max_steps_reached"
        # check if goal reached
        elif np.linalg.norm(np.array(self.box.position.tuple) - self.goal_pos) < RAD + GOAL_RAD:
            done = True
            reward = 0
            info["done"] = "goal reached"

        return self._get_obs(), reward, done, info

    def _get_obs(self):
        """ returns current state of environment """
        state = np.concatenate([self.pusher.position.tuple,
                                self.box.position.tuple,
                                self.pusher.linearVelocity.tuple,
                                self.box.linearVelocity.tuple,
                                self.goal_pos])
        return state

    def apply_hindsight(self, states, actions, goal_state):
        """ returns list of new states and list of new rewards for use with HER """
        goal = goal_state[2:4]  # get new goal location (last location of box)
        states.append(goal_state)
        num_tuples = len(actions)
        her_states, her_rewards = [], []
        states[0][-2:] = goal.copy()
        her_states.append(states[0])
        # for each state, adjust goal and calculate reward obtained
        for i in range(1, num_tuples + 1):
            state = states[i]
            state[-2:] = goal.copy()
            reward = self._HER_calc_reward(state)
            her_states.append(state)
            her_rewards.append(reward)
        return her_states, her_rewards

    def _HER_calc_reward(self, state):
        """ given state, returns reward for transitioning to this state (used by HER) """
        if np.linalg.norm(state[2:4] - state[4:6]) < RAD + GOAL_RAD:
            return 0
        else:
            return -1

    def set_state(self, state):
        self.pusher.position = state[:2]
        self.box.position = state[2:4]
        self.pusher.linearVelocity = state[4:6]
        self.box.linearVelocity = state[6:8]
        if len(state) == 10:  # The state can also be observation only, which does not include the goal
            self.goal_pos = state[8:10]

    def get_state(self):
        return copy.copy(self._get_obs())

    def get_nxt_state(self, state, action):
        original_state = self.get_state()
        original_elapsed_steps = self.elapsed_steps

        self.set_state(state)
        nxt_state, _, _, _ = self.step(action)
        nxt_state = nxt_state[:8]

        # Make sure there is no side effect
        self.set_state(original_state)
        self.elapsed_steps = original_elapsed_steps
        return nxt_state
