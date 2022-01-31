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


def general_render(env):
  L0 = 2*action_num - 1

  PoE = []

  for round in env.profile_history:
    round_PoE = 0;
    for strategy in round:
      for i in range(len(strategy)):
        delta_i = None
        if env.name == "DIR":
          delta_i = 2*env.mapping[i]-1
        round_PoE += strategy[i]*(delta_i/L0)
    PoE.append(round_PoE/len(round))

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

  anim = FuncAnimation(fig, animate, init_func=init, frames=range(0, len(env.profile_history), 1000), interval=1, blit=True)

  anim.save('DIR.gif', writer=PillowWriter(fps=10))

  plt.close()

class DIR(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self, std, num_players, num_actions, c = None):
    self.state = None
    self.std = std
    self.num_players = num_players
    self.num_actions = num_actions
    self.action = spaces.Discrete(num_actions)
    self.c = c if c != None else num_actions*2
    self.rho = max(self.c, self.num_actions)
    self.mapping = np.random.permutation(num_actions)
    print("Equilibrium action: " + str(self.mapping[num_actions-1]))
    self.profile_history = []
    self.name = "DIR"

  def step(self, actions):
    i = np.random.choice(range(self.num_actions), p=actions[0])
    j = np.random.choice(range(self.num_actions), p=actions[1])
    self.profile_history.append(actions)
    taken_actions = [i, j]
    i = self.mapping[i]
    j = self.mapping[j]
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
    done = bool(actions[0][self.mapping[self.num_actions-1]] == 1 or actions[1][self.mapping[self.num_actions-1]] == 1)
    #self.render()
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
    env.name = "SPA"
  
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
    env.name = "Lemon"

  def transform(self, x):
    return x*self.unit + self.minx

  def step(self, actions):
    self.profile_history.append(actions)
    taken_actions = []
    for i in range(len(actions)):
      taken_actions.append(np.random.choice(range(self.num_actions), p=actions[i]))

    actions = np.asarray(taken_actions)
    rewards = np.zeros(self.num_players)
    seller_actions = actions[1:]
    price =  self.transform( actions[0] ) - 1 

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
    pass


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