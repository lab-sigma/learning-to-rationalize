import numpy as np
import random, math
import scipy
from scipy.optimize import minimize_scalar, fsolve


strategy_choice_names = ['EXP3', 'EXP3P', 'EXP3Pswap', 'EXP3RVU', 'EXP3DH', 'OMDLogBarrier', 'MWU', 'MWUMB']

### EXP3 
class EXP3:
	def __init__(self, num_actions, num_iterations=None):
		self.num_actions = num_actions
		self.belief = np.ones(num_actions, dtype=float)
		self.belief /= sum(self.belief)
		self.gamma = 0
		self.action_prob = np.ones(num_actions, dtype=float) / num_actions
		self.t = 0
	def __str__(self):
		return f"EXP3\ngamma={self.gamma}\naction_prob={self.action_prob}\n" 

	def act(self, get_action_prob=False):
		self.t += 1
		self.gamma = min(1, math.sqrt( self.num_actions* 2 * math.log(self.num_actions) / (self.t) ) )

		self.action_prob = (1.0 - self.gamma) * self.belief + self.gamma / self.num_actions

		if get_action_prob:
			return self.action_prob

		action, = np.random.choice( self.num_actions, 1, p=self.action_prob )

		return action

	def feedback(self, action, reward, state=None):
		estimatedReward = reward / self.action_prob[action]
		self.belief[action] *= math.exp(estimatedReward * self.gamma / self.num_actions) 
		self.belief /= sum(self.belief)

	def get_action_prob(self):
		self.t += 1
		self.gamma = min(1, math.sqrt( self.num_actions* 2 * math.log(self.num_actions) / (self.t) ) )

		self.action_prob = (1.0 - self.gamma) * self.belief + self.gamma / self.num_actions

		return np.asarray(self.action_prob)


### EXP3.P
class EXP3P:
	def __init__(self, num_actions, num_iterations):
		self.num_actions = num_actions
		self.belief = np.ones(num_actions, dtype=float)
		self.belief /= sum(self.belief)
		self.gamma = 0
		self.action_prob = np.ones(num_actions, dtype=float) / num_actions
		self.t = 0
		self.num_iterations = num_iterations
		self.alpha = 2*np.sqrt( self.num_actions*self.num_iterations  )

	def __str__(self):
		return f"EXP3.P\ngamma={self.gamma}\nalpha={self.alpha}\naction_prob={self.action_prob}\n" 

	def act(self, get_action_prob=False):
		self.t += 1
		self.gamma = min(3/5, 2*math.sqrt( self.num_actions* 3/5 * math.log(self.num_actions) / (self.t) ) )

		self.action_prob = (1.0 - self.gamma) * self.belief + self.gamma / self.num_actions

		if get_action_prob:
			return self.action_prob

		action, = np.random.choice( self.num_actions, 1, p=self.action_prob )

		return action

	def feedback(self, action, reward, state=None):
		estimatedReward = (reward + self.alpha / np.sqrt(self.num_actions*self.num_iterations)) / self.action_prob[action]  
		self.belief[action] *= math.exp(estimatedReward * self.gamma / (3*self.num_actions) )  
		self.belief /= sum(self.belief)


### EXP3.P-swap - no internal regret
class EXP3Pswap:
	def __init__(self, num_actions, num_iterations):
		self.num_actions = num_actions
		self.action_prob = np.ones(num_actions, dtype=float) / num_actions
		self.t = 0
		self.players = [ EXP3P(self.num_actions, num_iterations) for i in range(self.num_actions) ]

	def __str__(self):
		return f"EXP3Pswap\naction_prob={self.action_prob}\n" 

	def act(self):
		Q = np.stack([p.act(get_action_prob=True) for p in self.players ], axis=0)
		val, vec = np.linalg.eig(Q.T)

		j_stationary = np.argmin(abs(val - 1.0))
		p_stationary = vec[:,j_stationary].real
		p_stationary /= p_stationary.sum()
		self.action_prob = p_stationary

		action, = np.random.choice( self.num_actions, 1, p=self.action_prob )

		return action

	def feedback(self, action, reward, state=None):
		estimatedReward = reward / self.action_prob[action]
		for i, p in enumerate(self.players):
			p.feedback(action, self.action_prob[i] * estimatedReward * p.action_prob[action] )

### EXP3-RVU  
class EXP3RVU:
	def __init__(self, num_actions, num_iterations=None):
		self.num_actions = num_actions
		self.belief = np.ones(num_actions, dtype=float)
		self.belief /= sum(self.belief)
		self.gamma = 0
		self.action_prob = []
		self.t = 0
		self.last_reward = None

	def __str__(self):
		return f"EXP3-RVU\ngamma={self.gamma}\naction_prob={self.action_prob}\n" 

	def act(self):
		self.t += 1
		self.gamma = min(1, math.sqrt( self.num_actions* 2 * math.log(self.num_actions) / (self.t) ) )

		### temporally add the extra reward of last round
		belief = self.belief.copy()
		if self.last_reward != None:
			(action, update) = self.last_reward
			belief[action] *= update
			belief /= sum(belief)

		self.action_prob = (1.0 - self.gamma) * belief + self.gamma / self.num_actions

		action, = np.random.choice( self.num_actions, 1, p=self.action_prob )

		return action

	def feedback(self, action, reward, state=None):
		estimatedReward = reward / self.action_prob[action]
		update = math.exp(estimatedReward * self.gamma / self.num_actions) 

		self.last_reward = (action, update)

		self.belief[action] *= update
		self.belief /= sum(self.belief)

### EXP3-DH
class EXP3DH:
	def __init__(self, num_actions, num_iterations=None, beta=None, b=0.2):
		self.num_actions = num_actions
		### not essential, but use higher precision just in case     
		self.loss = np.zeros(num_actions, dtype=np.longdouble)
		self.eps = 0
		self.action_prob = np.ones(num_actions, dtype=np.longdouble) / num_actions
		self.t = 0
		self.beta = 2 * num_actions if not beta else beta #or 1 for second price auction
		self.b = b

	def __str__(self):
		return f"EXP3-DH\nbeta={self.beta}\nb={self.b}\naction_prob={self.action_prob}\n"

	def act(self):
		action, = np.random.choice( self.num_actions, 1, p=self.action_prob.astype(float) )

		return action

	def feedback(self, action, reward, state=None):
		self.t += 1

		estimatedReward = reward / self.action_prob[action]
		discount = ( (self.t-1)/(self.t) ) ** (self.beta)
		self.eps = self.t ** (-self.b)
		self.loss *= discount
		self.loss[action] += estimatedReward

		### here the normalization through minus np.max(self.loss) 
		### is critical for maintain numerical stability 
		### while perserve the originial value
		exp_loss = np.exp( self.loss - np.max(self.loss) )

		self.action_prob =  (1-self.eps) * exp_loss / np.sum(exp_loss) + self.eps / self.num_actions
		

### OMD-LB
class OMDLogBarrier:
	def __init__(self, num_actions, num_iterations):
		self.num_actions = num_actions
		self.belief = np.ones(num_actions, dtype=float) / num_actions
		self.action_prob = self.belief
		self.t = 0

		self.T = num_iterations
		self.gamma = 1/self.T
		self.learning_rate = min( np.sqrt(num_actions/self.T), 1/(40*np.log(self.T)) )
		self.rho = np.ones(num_actions, dtype=float) * 2 * num_actions
		self.eta = np.ones(num_actions, dtype=float) * self.learning_rate
		self.kappa = np.exp(1/np.log(self.T))

	def __str__(self):
		return f"OMD-LB\neta={self.eta}\nkappa={self.kappa}\naction_prob={self.action_prob}\n"

	def act(self):
		self.belief = (1.0 - self.gamma) * self.belief + self.gamma / self.num_actions
		if abs(1-np.sum(self.belief) ) > 0.1 :
			print(self.belief, "the optimization program did not find the optimal value")
		self.belief = self.belief / np.sum(self.belief)

		self.action_prob = self.belief
		action, = np.random.choice( self.num_actions, 1, p=self.action_prob )

		return action

	def feedback(self, action, reward, state=None):
		estimatedLoss = (1-reward) / self.belief[action]
		losses = np.zeros(self.num_actions, dtype=float)
		losses[action] = estimatedLoss

		### solve for the normalization parameter of the OMD objective
		min_loss = max(0, np.min(losses))
		max_loss = np.max(losses)
		def objective(a_loss):
			lhs = np.sum(1. / ((1. / self.belief) + self.eta * (losses - a_loss)))
			rhs = 1.
			return (lhs - rhs) ** 2
		result = minimize_scalar(objective, bounds=(min_loss, max_loss), method='bounded')
		best_loss = result.x

		self.belief = 1. / ((1. / self.belief) + self.eta * (losses - best_loss))

		### increasing learning rate schedule
		for i in range(self.num_actions):
			if 1/self.belief[i] > self.rho[i]:
				self.rho[i] = 2. /self.belief[i]
				self.eta[i] = self.eta[i] * self.kappa				 

### MWU, as traditionally used with diminishing influence
class MWU:
	def __init__(self, num_actions, num_iterations=None, beta=None, b=0.2):
		self.num_actions = num_actions
		### not essential, but use higher precision just in case
		self.weights = np.ones(num_actions, dtype=np.longdouble)
		self.total_weight = num_actions
		self.eps = 1  # epsilon t, parameter that's decreasing with time. O(1/sqt(t))
		self.action_prob = np.ones(num_actions, dtype=np.longdouble) / num_actions
		self.t = 0

	def __str__(self):
		return f"MWU\naction_prob={self.action_prob}\n"

	def act(self):
		action, = np.random.choice( self.num_actions, 1, p=self.action_prob.astype(float) )

		return action

	def feedback(self, action, reward, state=None):
		self.t += 1

		self.eps = 1/((self.t+1) ** .5)

		estimatedReward = reward / self.action_prob[action]

		# Traditional MWU
		exp_update = np.exp( self.eps * estimatedReward )
		self.total_weight -= self.weights[action]
		self.weights[action] *= exp_update
		self.total_weight += self.weights[action]

		self.action_prob = ((1.0 - self.eps) * self.weights / self.total_weight) + (self.eps / self.num_actions)


### MWU-MB, as defined in Deng, et al to be Mean-Based
class MWUMB:
	def __init__(self, num_actions, num_iterations=None, beta=None, b=0.2):
		self.num_actions = num_actions
		### not essential, but use higher precision just in case
		self.weights = np.ones(num_actions, dtype=np.longdouble)
		self.total_weight = num_actions
		self.total_reward = np.zeros(num_actions, dtype=np.longdouble)
		self.eps = 1  # epsilon t, parameter that's decreasing with time. O(1/sqt(t))
		self.action_prob = np.ones(num_actions, dtype=np.longdouble) / num_actions
		self.t = 0

	def __str__(self):
		return f"MWU\naction_prob={self.action_prob}\n"

	def act(self):
		action, = np.random.choice( self.num_actions, 1, p=self.action_prob.astype(float) )

		return action

	def feedback(self, action, reward, state=None):
		self.t += 1

		self.eps = 1/((self.t+1) ** .5)
		estimatedReward = reward / self.action_prob[action]
		
		# Deng, et al
		self.total_reward[action] += estimatedReward
		self.weights = np.exp(self.eps * self.total_reward)
		self.total_weight = np.sum(self.weights)

		self.action_prob = ((1.0 - self.eps) * self.weights / self.total_weight) + (self.eps / self.num_actions)



