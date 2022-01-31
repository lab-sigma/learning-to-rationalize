import argparse,time,os,pickle
import numpy as np
import os

from player import *
from env import *
from utils import * 

np.set_printoptions(precision=2)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	describe = lambda names :  ''.join( ['{}: {},'.format(i, n) for i,n in enumerate(names)] )

	parser.add_argument('--std', type=float, default=0, help='noise std. in feedback')
	parser.add_argument('--iterations', type=int, default=100, help='number of rounds to play')
	parser.add_argument('--strategy', type=int, help='strategies: ' + describe(strategy_choice_names))
	parser.add_argument('--environment', type=int, default=0, help='environments: ' + describe(env_choice_names) )
	parser.add_argument('--num_players', type=int, default=2, help='number of players ' )
	parser.add_argument('--num_actions', type=int, help='number of actions ')
	parser.add_argument('--unit', type=float, default=None, help='discretized unit')
	parser.add_argument('--minx', type=float, default=0, help='min action')
	parser.add_argument('--samples', type=int, default=100, help='number of samples to save' )
	parser.add_argument('--new', default=False, action='store_true', help='whether to generate a new env instance')
	parser.add_argument('--num_repeat', type=int, default=1, help='number of repeated simulation')
	parser.add_argument('--use', type=int, default=-1, help='use a specified env instance, use -1 for the latest instance')
	parser.add_argument('--values', type=int, nargs='+', default=None, help='space separated sequence of integer valuations, range [0, num_actions), size num_players')

	args = parser.parse_args()

	std = args.std
	num_iterations = args.iterations
	strategy = args.strategy

	num_players = args.num_players
	num_actions = args.num_actions

	unit = (1.0/num_actions) if (args.unit is None) else args.unit
	minx = args.minx

	samples = args.samples
	values = np.array(args.values)


	env_choice = args.environment
	env_name = env_choice_names[env_choice]
	strategy_name = strategy_choice_names[strategy]

	j = 0
	while j < args.num_repeat:
		# log_dir = f'results/{env_name}/{strategy_name}'
		log_dir = os.path.join("results", env_name, strategy_name)
		if not os.path.exists(log_dir):
		    os.makedirs(log_dir)
		    print("created directory")
		else:
			print("existing directory")

		prefix = os.path.join("results", env_name, str(num_actions)+"_"+str(num_players)+"_"+str(std)+"_"+str(unit)+"_"+str(minx))
		i = find_latest(prefix, '.pickle')
		if args.new or i==0: ### create new environment instance 
			env_module = __import__('env')
			env = getattr(env_module, env_name)(std, num_actions,num_players, unit, minx, values=values)

			env_dir = prefix + str(i) + '.pickle' 
			f = open(env_dir, 'wb')
			pickle.dump(env, f )
			print("save env at "+ env_dir)
			f.close()
		else: ### load existing environment instance
			if args.use == -1: ### use the last env instance by default
				i -= 1
			else:
				i = args.use

			env_dir = prefix + str(i) + '.pickle' 
			if not os.path.exists(log_dir):
			    print("env path not found ", log_dir)
			    exit()
			f = open(env_dir, 'rb')
			env = pickle.load(f)
			print("load env at " + env_dir)
			f.close()

		player_module = __import__('player')
		players = [getattr(player_module, strategy_name)(num_actions, num_iterations) for i in range(num_players) ]
	    
		i = find_latest(os.path.join(log_dir,), ".log")
		log_dir = os.path.join(log_dir, str(i))

		L = logger(log_dir, env, num_iterations, samples=samples, player=strategy_name)
		start = time.time()
		L.write(f"iterations: {num_iterations}\n")
		L.write(f"Environment:\n\t{env}\n")

		actions = np.zeros(num_players, dtype=int)

		### start the simulation
		for t in range(1, num_iterations+1):

			for i, p in enumerate(players): 
				actions[i] = p.act()

			### send action profile and get payoff from the environment
			rewards = env.feedback( actions )

			### send payoff to each players
			for a, p, r in zip(actions, players, rewards ):
				p.feedback(a, r)

			L.record_round(t, actions, rewards, players)

		for i, p in enumerate(players): 
			L.write(f'Player{i}:\n\t{p}\n')

		L.plot()

		end = time.time()
		print(log_dir, end-start)

		j += 1
