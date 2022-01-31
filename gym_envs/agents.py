class EXP3:
	def __init__(self, num_actions, num_iterations=None):
		self.num_actions = num_actions
		self.belief = np.ones(num_actions, dtype=longdouble)
		self.belief /= sum(self.belief)
		self.gamma = 0
		self.action_prob = np.ones(num_actions, dtype=longdouble) / num_actions
		self.t = 0

	def __str__(self):
		return f"EXP3\ngamma={self.gamma}\naction_prob={self.action_prob}\n"

	def feedback(self, action, reward, state=None):
		self.t += 1

		estimatedReward = reward / self.action_prob[action]
		self.belief[action] *= math.exp(estimatedReward * self.gamma / self.num_actions)
		self.belief /= sum(self.belief)

		self.gamma = min(1, math.sqrt(self.num_actions * 2 * math.log(self.num_actions) / (self.t)))
		self.action_prob = (1.0 - self.gamma) * self.belief + self.gamma / self.num_actions


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


# Mean-based multiplicative weight update from Deng, et al
class MWUMB:
	def __init__(self, num_actions, num_iterations=None):
		self.num_actions = num_actions
		### not essential, but use higher precision just in case
		self.weights = np.ones(num_actions, dtype=np.longdouble)
		self.total_weight = num_actions
		self.total_reward = np.zeros(num_actions, dtype=np.longdouble)
		self.eps = 1  # epsilon t, parameter that's decreasing with time. O(1/sqt(t))
		self.action_prob = np.ones(num_actions, dtype=np.longdouble) / num_actions
		self.t = 0

	def __str__(self):
		return f"MWU-MB\naction_prob={self.action_prob}\n"

	def feedback(self, action, reward, state=None):
		self.t += 1

		self.eps = 1 / ((self.t + 1) ** .5)
		estimatedReward = reward / self.action_prob[action]

		# Deng, et al
		self.total_reward[action] += estimatedReward
		self.weights = np.exp(self.eps * self.total_reward)
		self.total_weight = np.sum(self.weights)

		self.action_prob = ((1.0 - self.eps) * self.weights / self.total_weight) + (self.eps / self.num_actions)