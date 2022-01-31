import argparse,time,os,pickle

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def find_latest(prefix, suffix):
	i = 0
	while os.path.exists(f'{prefix}{i}{suffix}'): 
		i += 1	
	return i

class logger:
	def __init__(self, log_dir, env, num_iterations, samples=None, player=""):
		self.log_dir = log_dir
		self.env = env
		self.hist_action_freq = np.zeros( (env.num_players, env.num_actions) )
		self.num_iterations = num_iterations
		self.player = player

		self.samples = self.num_iterations if not samples else samples
		self.step_size = self.num_iterations // self.samples

		### store the selected action by each player sampled at every step_size round
		self.sampled_action = np.zeros( (env.num_players, self.samples) ) 
		### store action probability of each player sampled at every step_size round
		self.belief_history = [] 
		### store the frequency of the selected actions by each player from from round i*step_size to (i+1)*step_size
		self.action_history = [] 

	def write(self, text):
		with open(self.log_dir + '.log', 'a') as f:
			f.write(text)	

	def record_round(self, t, actions, rewards, players):
		for i, a in enumerate(actions):
			self.hist_action_freq[i][a] += 1 

		if t % self.step_size == 0:
			its = t // self.step_size - 1
			for i, a in enumerate(actions):
				self.sampled_action[i][its] = self.env.transform_action(a) 
			
			self.action_history.append( self.hist_action_freq / self.step_size )
			self.hist_action_freq.fill(0)

			beliefs = []
			for p in players:
				beliefs.append(p.action_prob)
			self.belief_history.append(beliefs)



	def plot_action_history(self): ### a draft plot for the action profile trend
		time_axis = np.arange(0, self.num_iterations, step=self.step_size) 

		fig, ax = plt.subplots()
		for i in reversed(range(self.env.num_players)):
			ax.plot(time_axis, self.sampled_action[i], label=f"P{i}", lw=1)
			# ax.scatter(time_axis, self.sampled_action[i], label=f"P{i}")
		ax.set_ylabel('Bids')
		ax.set_xlabel('#round')
		ax.set_title(str(self.player) + " " + str(self.env))
		ax.legend(loc="lower right")
		plt.savefig(self.log_dir + '_action_history' '.png')
		plt.clf()

	def plot(self):
		self.plot_action_history()

		with open(self.log_dir + '_history.pickle', 'wb') as f:
			pickle.dump((self.action_history, self.sampled_action, self.belief_history), f)



