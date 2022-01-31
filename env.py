import numpy as np


env_choice_names = ['DIR', 'SPA', 'FPA']


class base_env:
	def __init__(self, std, num_actions, num_players, unit, minx):
		self.std = std
		self.unit = unit
		self.minx = minx
		self.num_players = num_players
		self.num_actions = num_actions

	def transform_action(self, actions): ### by default not to transform actions
		return actions 

	def feedback(self, actions):
		pass

### the "Diamond in the Rough" game with bandit feedback
class DIR(base_env):
	def __init__(self, std, num_actions, num_players, unit, minx, c=None):
		assert num_players == 2,  "DIR game is currently designed for 2 players"
		base_env.__init__(self, std, num_actions, num_players, unit, minx)

		self.c = c if c != None else num_actions*2
		self.rho = max(self.c, self.num_actions)

	def __str__(self):
		return f"DIR({self.num_actions}, {self.c}) with noise std. {self.std}\n"

	def feedback(self, actions):
		i = actions[0]+1
		j = actions[1]+1
		rewards = np.zeros(self.num_players, dtype=float)

		if i <= j+1:
			rewards[0] = i/self.rho
		else:
			rewards[0] = -self.c/self.rho

		if j <= i:
			rewards[1] = j/self.rho
		else:
			rewards[1] = -self.c/self.rho

		### apply iid gaussian noise to the payoff
		rewards += np.random.randn(self.num_players) * self.std
		
		return rewards

### the repeated second price auction game with bandit feedback
class SPA(base_env):
	def __init__(self, std, num_actions, num_players, unit, minx, values=None):
		base_env.__init__(self, std, num_actions, num_players, unit, minx)
		### randomly sample values for players and wlog rank players by its value 
		self.values = minx + np.sort( np.random.choice(num_actions, num_players) if (values is None) else values )*unit

	def __str__(self):
		return f"SPA({self.num_players}, {self.num_actions}) with noise std. {self.std}\nValues {self.values}"

	def transform_action(self, actions):
		return self.minx + actions*self.unit ### to linearly map an action id to a real value

	def feedback(self, actions):
		bids = self.transform_action(actions)
		w = np.argmax(bids)
		bids[w] = -1 ## assume all positive bid
		price = np.max(bids)

		rewards = np.zeros(self.num_players)
		noise = np.random.randn() * self.std
		### noisy feedback for the winner
		rewards[w] = self.values[w] - price + noise

		return rewards


### the repeated first price auction game with bandit feedback
class FPA(base_env):
	def __init__(self, std, num_actions, num_players, unit, minx, values=None):
		base_env.__init__(self, std, num_actions, num_players, unit, minx)
		### randomly sample values for players and wlog rank players by its value
		self.values = minx + np.sort( np.random.choice(num_actions, num_players) if (values is None) else values )*unit

	def __str__(self):
		return f"FPA({self.num_players}, {self.num_actions}) with noise std. {self.std}\nValues {self.values}"

	def transform_action(self, actions):
		return self.minx + actions*self.unit ### to linearly map an action id to a real value

	def feedback(self, actions):
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

		return rewards
