# Simulation program for the experiments of STOC 2022 submission
### *Multi-Agent  Learning for  Iterative Dominance Elimination: \\ Formal Barriers and New Algorithms*

![DIR(20,40) animation](DIR(20,40).gif)

![DIR(10,20) animations](DIR(10,20).gif)

```
### get help on how to specify simulation parameter
python main.py -h

### to start a DIR(20, 40) game simulation of 2 EXP3-DH players
python main.py --num_actions 20 --environment 0 --strategy 4 --iterations 100000000 --std 0.1

### to start an SPA game simulation of 10 EXP3 players and 20 action in [0, 1)
python main.py --num_players 10 --num_actions 20 --unit 0.05 --minx 0 --environment 1 --strategy 0 --iterations 100000 --std 0.1

### to start an Lemon game simulation of 50 EXP3 sellers and 1 EXP3 buyers with 50 action 
python lemon.py --num_sellers 50  --num_actions 50 --unit 1 --minx 25 --environment 1 --strategy 0 --iterations 10000 --std 0.1

```
