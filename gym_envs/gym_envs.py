import gym
import numpy as np
from gym import spaces
import random, math
import scipy
from scipy.optimize import minimize_scalar, fsolve
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def general_render(de_hist, game_name, num_players, num_actions, std, steps):

  fig, ax1 = plt.subplots()

  line1, = ax1.plot([], [], linestyle = '-', color='tab:blue', label="EXP3-DH",  marker='o')

  ax1.set_title("EXP3-DH")#y=title_pos, fontsize=title_fontsize, fontweight=fontweight)

  txt = fig.text(0.45, 0.9, s="placeholder")

  def update(i):
    j = max(0, i-100)
    line1.set_data(de_hist[0][j:i], de_hist[1][j:i])
    txt.set_text(f"$t={i+1}^3$")
    return ax1

  fig.suptitle(f'Learning Dynamics in {game_name}({player_num}, {action_num}) w/ noise {std}')

  anim = FuncAnimation(fig, update, frames=np.arange(steps), interval=50)
  anim.save('learning_dynamics' + '.gif', dpi=80, writer='ffmpeg')

class DIR(gym.Env):
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
    return rewards, taken_actions, done


  def reset(self):
    pass

  def render(self):
    pass

class SPA(gym.Env):
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
    bids = self.transform_action(taken_actions)
    price = np.max(bids)

    winners = (bids == price)
    num_winners = np.count_nonzero(winners)

    noise = np.random.randn(self.num_players) * self.std
    ### noisy feedback for the winner
    rewards = (((self.values - price) / num_winners) + noise) * winners

    return rewards, taken_actions

  def render(self):
    pass

  def reset(self):
    pass