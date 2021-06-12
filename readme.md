# Simulation program for the experiments of neurips-21 submission
### *Multi-Agent  Learning for  Iterative Dominance Elimination ---  the Merit of Forgetting*


```
### get help on how to specify simulation parameter
python main.py -h

### to start a DIR(20, 40) game simulation of 2 EXP3-DH players
python main.py --num_actions 20 --environment 0 --strategy 4 --iterations 100000000 --std 0.1

### to start an SPA game simulation of 10 EXP3 players and 20 action in [0, 1)
python main.py --num_players 10 --num_actions 20 --unit 0.05 --minx 0 --environment 1 --strategy 0 --iterations 100000 --std 0.1



```