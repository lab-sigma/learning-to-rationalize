import argparse,time,os,pickle
import matplotlib.pyplot as plt
import numpy as np
from player import *
plt.switch_backend('agg')


np.set_printoptions(precision=2)

class lemon:
	def __init__(self, std, num_sellers, num_actions, unit, minx):
		self.std = std
		self.unit = unit
		self.num_sellers = num_sellers
		self.num_players = num_sellers + 1
		self.quality = self.transform(np.arange(num_sellers) )

		self.num_actions = num_actions
		self.welfare_factor = 1.5
		self.listing_cost = 3

	def __str__(self):
		return f"Lemon({self.num_sellers}) with noise std. {self.std},\nquality: {self.quality}\n"

	def transform(self, x):
		return x*unit + minx

	def feedback(self, actions):
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

		return rewards, supply, price, avg_quality

class logger:
	def __init__(self, log_dir, env, iterations, samples=None):
		self.log_dir = log_dir
		self.env = env
		self.supply_history = []
		self.demand_history = []
		self.price_history = []
		self.avg_quality_history = []

		self.iterations = iterations
		self.samples = self.iterations if not samples else samples
		self.step_size = self.iterations // self.samples

		self.sampled_actions = []

	def write(self, text):
		with open(self.log_dir+ '.log', 'a') as f:
			f.write(text)	

	def record_round(self, t, supply, price, avg_quality, actions):
		if t % self.step_size == 0:
			self.supply_history.append(supply)
			self.price_history.append(price)
			self.avg_quality_history.append(avg_quality)
			self.sampled_actions.append(actions[1:].copy())


	def plot(self):
		time_axis = np.arange(0, self.iterations, step=self.step_size) 

		fig, (ax1, ax2) = plt.subplots(2, 1)
		ax1.plot(time_axis, self.supply_history, label=f"supply")
		ax1.set_ylabel('#units')
		ax1.legend(loc="upper left")

		ax2.plot(time_axis, self.price_history, label=f"price")
		ax2.plot(time_axis, self.avg_quality_history, label=f"avg. quality")
		ax2.set_ylabel('$')
		ax2.set_xlabel('#round')
		ax2.legend(loc="upper left")

		fig.suptitle( f"Lemon({self.env.num_sellers}) with noise std. {self.env.std}")
		plt.savefig(self.log_dir+ '_price' '.png')
		plt.clf()

		fig, ax3 = plt.subplots(1, 1)
		im = ax3.imshow(np.asarray( self.sampled_actions).T, aspect="auto")
		cbar = ax3.figure.colorbar(im, ax=ax3)
		cbar.ax.set_ylabel("prob. to sell", rotation=-90, va="bottom")

		ax3.set_yticks(np.arange(0, self.env.num_sellers, step=5))
		ax3.set_ylabel('#player')
		ax3.set_xlabel('#round')

		fig.suptitle( f"Lemon({self.env.num_sellers}) with noise std. {self.env.std}")
		plt.savefig(self.log_dir+ '_trend' '.png')
		plt.clf()

		with open(self.log_dir+'_history.pickle', 'wb') as f:
			pickle.dump(self.sampled_actions, f)

def find_latest(prefix, suffix):
	i = 0
	while os.path.exists(f'{prefix}{i}{suffix}'): 
		i += 1	
	return i

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	describe = lambda names :  ''.join( [', {}: {}'.format(i, n) for i,n in enumerate(names)] )

	parser.add_argument('--std', type=float, default=0, help='noise std. in feedback')
	parser.add_argument('--iterations', type=int, default=100, help='number of rounds to play')
	parser.add_argument('--strategy', type=int, help='player strategy' + describe(player_strategy_names))
	parser.add_argument('--num_sellers', type=int, help='number of sellers ' )
	parser.add_argument('--num_actions', type=int, help='number of buyers ')
	parser.add_argument('--unit', type=float, default=1, help='discretized unit')
	parser.add_argument('--minx', type=float, default=0, help='min action')
	parser.add_argument('--samples', type=int, default=100, help='number of samples to save' )
	parser.add_argument('--new', default=False, action='store_true', help='whether to generate a new env instance')
	parser.add_argument('--num_repeat', type=int, default=1, help='number of repeated simulation')
	parser.add_argument('--force_env', default=False, action='store_true', help='whether to use a specified env instance')

	args = parser.parse_args()

	std = args.std
	iterations = args.iterations
	strategy = args.strategy

	num_sellers = args.num_sellers
	num_buyers = 1
	num_actions = args.num_actions
	num_players = num_sellers+num_buyers

	unit = args.unit
	minx = args.minx

	samples = args.samples

	env_name = "lemon3"
	strategy_name = player_strategy_names[strategy]


	j = 0
	while j < args.num_repeat:
		log_dir = f'results/{env_name}/{strategy_name}'
		if not os.path.exists(log_dir):
		    os.makedirs(log_dir)
		    print("created directory")
		else:
			print("existing directory")

		prefix = f'results/{env_name}/{num_sellers}_{num_buyers}|{std}|{unit}|{minx}#'
		if not args.force_env:
			i = find_latest(prefix, '.pickle')
			if not args.new and i > 0:
				env_dir = prefix + str(i-1) + '.pickle' 
				f = open(env_dir, 'rb')
				env = pickle.load(f)
				print("load env at " + env_dir)
				f.close()
			else:
				env = lemon(std, num_sellers, num_actions, unit, minx)

				env_dir = prefix + str(i) + '.pickle' 
				f = open(env_dir, 'wb')
				pickle.dump(env, f )
				print("save env at "+ env_dir)
				f.close()
		else:
			i = specified_env[j]
			env_dir = prefix + str(i) + '.pickle' 
			if not os.path.exists(log_dir):
			    print("env path not found ", log_dir)
			    exit()
			f = open(env_dir, 'rb')
			env = pickle.load(f)
			print("load env at " + env_dir)
			f.close()

		if strategy != 4:
			players = [player(strategy, num_actions, iterations) ]
			players.extend( [player(strategy, 2, iterations) for i in range(num_sellers) ] )
		else:
			a0 = 50
			b0 = 0.5
			a1 = 50
			b1 = 0.5
			players = [DEplayer(num_actions, a0, b0) ]
			players.extend( [DEplayer(2, a1, b1) for i in range(num_sellers) ] )
			print(f'beta = {players[0].a}, b = {players[0].b}, beta = {players[1].a}, b = {players[1].b}' )
	    
		i = find_latest(f'{log_dir}/', '.log')
		log_dir = f'{log_dir}/{i}'

		L = logger(log_dir, env, iterations, samples=samples)
		start = time.time()
		L.write("iterations: "+str(iterations) + "\n")
		L.write('Environment:\n\t'+str(env)+'\n')

		actions = np.zeros(num_players, dtype=int)
		action_probs = np.zeros(num_players, dtype=float)


		for t in range(1, iterations+1):

			for i, p in enumerate(players): 
				actions[i] = p.act( None )
				action_probs[i] = p.action_prob[1]

			rewards, supply, price, avg_quality = env.feedback( actions )

			for a, p, r in zip(actions, players, rewards ):
				p.feedback(a, r)

			L.record_round(t, supply, price, avg_quality, action_probs)

		for i, p in enumerate(players): 
			L.write(f'Player{i}:\n\t{p}\n')

		L.plot()

		end = time.time()
		print(log_dir, end-start)

		j += 1
