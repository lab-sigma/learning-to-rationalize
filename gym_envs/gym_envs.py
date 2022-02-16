import gym
import numpy as np
from gym import spaces
import random, math
import scipy
from scipy.optimize import minimize_scalar, fsolve
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation, PillowWriter
import sys
matplotlib.use('agg')

def compute_PoE(env):
  name = env.name
  PoE = []
  print_check = True
  #"Round" here corresponds to the set of action profiles for each agent on a given round
  for round_num, round in enumerate(env.profile_history):
    #This calculates the PoE for the Market for Lemons game by simply adding up the probabilities for each seller of selling
    if env.name == "Lemon":
      round_PoE = 0
      for agent, action_profile in enumerate(round[1:]):
        if round_num % 100 == 0:
          print(f"Action profile for agent {agent+1}: {action_profile}")
        round_PoE += action_profile[env.mappings[agent+1][0]]
      PoE.append(round_PoE/env.num_sellers)
      if round_num % 100 == 0:
        print(f"Total PoE: {round_PoE/env.num_sellers}")
    #This calculates the PoE for the DIR game by implementing the formula seen on page 12 of https://arxiv.org/pdf/2111.05486.pdf
    if env.name == "DIR":
      L0 = 2*env.num_actions - 2
      round_PoE = 0
      for agent, action_profile in enumerate(round):
        for action, action_prob in enumerate(action_profile):
          delta_i = None
          if env.name == "DIR":
            if agent == 0:
              Lambda_i = 2*env.mappings[agent][action]
            else:
              Lambda_i = 2*env.mappings[agent][action] + 1
              #Edge case to ensure that Lambda_{n-1} = 2*num_actions
              if env.mappings[agent][action] == env.num_actions-1:
                Lambda_i = 2*(env.num_actions-1)
          round_PoE += action_prob*(Lambda_i/L0)
      if round_num % 100000 == 0:
        print(f"Agent 0's action profile: {env.profile_history[round_num][0]} for round {round_num}")
        print(f"Agent 1's action profile: {env.profile_history[round_num][1]} for round {round_num}")
        print(f"Round PoE: {round_PoE/env.num_players}")
      if round_PoE/env.num_players > 1 and print_check:
        print(round_PoE/env.num_players)
        print_check = False
      PoE.append(round_PoE/env.num_players)
  return PoE

def general_render(env):

  PoE = compute_PoE(env)

  fig = plt.figure()
  ax = plt.axes(xlim=(0, len(env.profile_history)), ylim=(0, 1))

  ax.set_xlabel('Round Number')
  ax.set_ylabel('Progress of Elimination')
  if env.name == "DIR":
    ax.set_title(f'Progress of Elimination for {env.name}({env.num_actions}, {env.c})')
  else:
    ax.set_title(f'Progress of Elimination for {env.name}({env.num_actions}, {env.num_players})')

  line, = ax.plot([], [], lw = 1)

  def init():
    line.set_data([], [])
    return line,

  def animate(i):
    x = range(0, i)
    y = PoE[:i]
    line.set_data(x, y)
    return line,

  anim = FuncAnimation(fig, animate, init_func=init, frames=range(0, len(env.profile_history), len(env.profile_history)//100), interval=1, blit=True)

  anim.save(f'{env.name}({env.num_actions}, {env.num_players})_for_{len(env.profile_history)}.gif', writer=PillowWriter(fps=10))

  plt.close()

class DIR(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self, std, num_players, num_actions, c = None):
    self.state = None
    self.std = std
    self.num_players = num_players
    self.num_actions = num_actions
    self.c = c if c != None else num_actions*2
    self.rho = max(self.c, self.num_actions)
    self.mappings = []
    #mappings represents the random permutations of the action space for each agent to ensure the equilibrium is different on each run
    for i in range(num_players):
      self.mappings.append(np.random.permutation(num_actions))
    self.mapping = np.random.permutation(num_actions)
    print(f"Equilibrium actions: {self.mappings[0][num_actions-1]}, {self.mappings[1][num_actions-1]}")
    #profile_history stores the set of action profiles for each agent
    self.profile_history = []
    self.name = "DIR"

  def step(self, actions):
    #randomly choose an action based on the set of action profiles for each agent
    i = np.random.choice(range(self.num_actions), p=actions[0])
    j = np.random.choice(range(self.num_actions), p=actions[1])
    #save the set of action profiles to profile_history
    self.profile_history.append(actions)
    taken_actions = [i, j]
    #randomly map the chosen actions
    i = self.mappings[0][i]
    j = self.mappings[1][j]
    rewards = np.zeros(2, dtype=float)
    if i <= j+1:
      rewards[0] = i/self.rho
    else:
      rewards[0] = -self.c/self.rho
    if j <= i:
      rewards[1] = j/self.rho
    else:
      rewards[1] = -self.c/self.rho
    rewards += np.random.randn(2) * self.std
    return rewards, taken_actions

  def reset(self):
    pass

  def render(self):
    general_render(self)

class SPA(gym.Env):
  metadata = {'render.modes': ['human']}
  def __init__(self, std, num_actions, num_players, unit, minx):
    self.state = None
    self.std = std
    self.num_players = num_players
    self.minx = minx
    self.unit = unit
    self.num_actions = num_actions
    self.action = spaces.Discrete(num_actions)
    ### randomly sample values for players and wlog rank players by its value 
    self.values = minx + np.sort( np.random.choice(num_actions, num_players) )*unit
    self.profile_history = []
    self.name = "SPA"
  
  def transform_action(self, actions):
    return self.minx + actions*self.unit ### to linearly map an action id to a real value

  def step(self, actions):
    self.profile_history.append(actions)
    taken_actions = []
    for i in range(len(actions)):
      taken_actions.append(np.random.choice(range(self.num_actions), p=actions[i]))
    taken_actions = np.asarray(taken_actions)
    bids = self.transform_action(taken_actions)
    w = np.argmax(bids)
    bids[w] = -1 ## assume all positive bid
    price = np.max(bids)

    rewards = np.zeros(self.num_players)
    noise = np.random.randn() * self.std
		### noisy feedback for the winner
    rewards[w] = self.values[w] - price + noise
    return rewards, taken_actions

  def render(self):
    pass

  def reset(self):
    pass

class Lemon(gym.Env):
  #For PoE calculation, add up probability of selling for each agent divided by the number of agents that are selling
  metadata = {'render.modes': ['human']}
  def __init__(self, std, num_sellers, num_actions, unit, minx):
    self.std = std
    self.unit = unit
    self.minx = minx
    self.num_sellers = num_sellers
    self.num_players = num_sellers + 1
    self.quality = self.transform(np.arange(num_sellers))
    self.num_actions = num_actions
    self.welfare_factor = 1.5
    self.listing_cost = 3
    self.profile_history = []
    #mappings represents the random permutations of the action space for each agent to ensure the equilibrium is different on each run
    self.mappings = []
    for i in range(self.num_players):
      self.mappings.append(np.random.permutation(2))
    #set mapping for buyer to be different from that of sellers
    self.mappings[0] = np.random.permutation(num_actions)
    #profile_history stores the set of action profiles for each agent
    self.name = "Lemon"

  def transform(self, x):
    return x*self.unit + self.minx

  def step(self, action_profiles):
    self.profile_history.append(action_profiles)
    actions = []

    #choose actions randomly from action_profile
    actions.append(np.random.choice(range(self.num_actions), p=action_profiles[0]))
    for i, action_profile in enumerate(action_profiles[1:]):
      actions.append(np.random.choice(range(2), p=action_profile))

    taken_actions = actions.copy()

    #swap action to randomly mapped action
    for i, action in enumerate(actions):
      actions[i] = self.mappings[i][action]

    actions = np.asarray(actions)
    rewards = np.zeros(self.num_players)
    seller_actions = actions[1:]
    price =  self.transform(actions[0]) - 1 

    sold = seller_actions * (self.quality < price) ### quality below price and is selling
    supply = np.sum(sold)
    if supply > 0:
      avg_quality = np.sum(sold * self.quality) / supply
      q_noise = np.random.randn(self.num_sellers) * 5
      rewards[1:] = seller_actions * [ (self.quality + q_noise < price) * (price - self.quality) - self.listing_cost ]
      rewards[0] = ( self.welfare_factor * avg_quality - price ) 

      noise = np.random.randn(self.num_players) * self.std
      rewards += noise

    else:
      avg_quality = 0 
      rewards = np.zeros(self.num_players)
      rewards[1:] = - seller_actions * self.listing_cost 
    rewards /= self.num_players

    return rewards, taken_actions

  def reset(self):
    pass

  def render(self):
    general_render(self)


# repeated first price auction
class FPA(gym.Env):
  def __init__(self, std, num_actions, num_players, unit, minx, values=None):
    self.state = None
    self.std = std
    self.num_players = num_players
    self.minx = minx
    self.unit = unit
    self.num_actions = num_actions
    self.action = spaces.Discrete(num_actions)
    ### randomly sample values for players and wlog rank players by its value
    self.values = minx + np.sort( np.random.choice(num_actions, num_players) if (values is None) else values )*unit
    self.profile_history = []

  def transform_action(self, actions):
    return self.minx + actions * self.unit  ### to linearly map an action id to a real value

  def step(self, actions):
    self.profile_history.append(actions)
    taken_actions = []
    for i in range(len(actions)):
      taken_actions.append(np.random.choice(range(self.num_actions), p=actions[i]))
    taken_actions = np.asarray(taken_actions)

    bids = self.transform_action(actions)
    # w = np.argmax(bids)
    # bids[w] = -1 ## first prize auction, we don't need to find second highest
    price = np.max(bids)

    winners = (bids == price)
    # nonwinners = (bids != price)
    num_winners = np.count_nonzero(winners)

    # bidders = (bids != 0)   # make the trajectories cyclic, 0 = abstain from bidding
    bidders = np.ones(self.num_players)  # make penalty uniform, no one abstains from bidding

    # noise = np.random.randn(self.num_players) * self.std
    noise = np.random.randn() * self.std
    ### noisy feedback for the winner
    rewards = (((self.values + noise - price) / num_winners) * winners) - (self.unit * bidders)

    return rewards, taken_actions

  def render(self):
    pass

  def reset(self):
    pass