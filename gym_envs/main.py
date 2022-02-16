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

def run_env(env, num_iterations):
  agents = []
  for i in range(env.num_players):
    if env.name == "DIR":
      agents.append(EXP3DH(env.num_actions))
    if env.name == "Lemon":
      if i == 0:
        agents.append(EXP3DH(env.num_actions))
      else:
        agents.append(EXP3DH(2))

  for t in range(num_iterations):
    actions = []
    for i in range(env.num_players):
      actions.append(agents[i].action_prob.astype(float))
    rewards, taken_actions = env.step(actions)
    for i in range(env.num_players):
      agents[i].feedback(taken_actions[i], rewards[i])

  for i in range(env.num_players):
    print("Agent ", i, "'s final distribution over actions: ", agents[i].action_prob.astype(float))

num_iterations = 10000
action_num = 24
player_num = 25
#env = DIR(num_actions=action_num, std = .1, num_players=2)
#env = SPA(num_actions=action_num, std = .1, num_players=player_num, unit = .05, minx = 0)
env = Lemon(num_actions = action_num, unit = 1, minx = 25, std=.1, num_sellers=player_num-1)
agents = []

run_env(env, num_iterations)

env.render()