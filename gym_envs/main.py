import gym
import numpy as np
from gym import spaces
import random, math
import scipy
from scipy.optimize import minimize_scalar, fsolve
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from gym_envs import DIR, SPA, Lemon
from agents import EXP3DH

num_iterations = 100000
action_num = 5
player_num = 2
env = DIR(num_actions=action_num, std = .1, num_players=2)
#env = SPA(num_actions=action_num, std = .1, num_players=player_num, unit = .05, minx = 0)
#env = Lemon(num_actions = action_num, unit = 1, minx = 25, std=.1, num_sellers=player_num-1)
#env = FPA(num_actions=action_num, std = .1, num_players=player_num, unit = .05, minx = 0)
agents = []

for i in range(player_num):
  # agents.append(EXP3(action_num))
  agents.append(EXP3DH(action_num))
  # agents.append(MWUMB(action_num))

for t in range(num_iterations):
  actions = []
  for i in range(player_num):
    actions.append(agents[i].action_prob.astype(float))
  rewards, taken_actions = env.step(actions)
  for i in range(player_num):
    agents[i].feedback(taken_actions[i], rewards[i])

for i in range(player_num):
  print("Agent ", i, "'s final distribution over actions: ", agents[i].action_prob.astype(float))

env.render()