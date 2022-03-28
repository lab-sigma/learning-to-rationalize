import os
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
from gym_envs.gym_envs import DIR, SPA, Lemon
from gym_envs.gym_envs import compute_PoE
from gym_envs.agents import EXP3, EXP3DH, MWUMB

from aicrowd_gym.servers.zmq_oracle_server import ZmqOracleServer
from aicrowd_gym.serializers import MessagePackSerializer


class Constants:
    SERVER_HOST = os.getenv("AICROWD_REMOTE_SERVER_HOST", "0.0.0.0")
    SERVER_PORT = int(os.getenv("AICROWD_REMOTE_SERVER_PORT", "5000"))

class EXP3DHAgent:
  def __init__(self):
    self.num_actions = 2
		### not essential, but use higher precision just in case     
    self.loss = np.zeros(self.num_actions, dtype=np.float128)
    self.eps = 0
    self.action_prob = np.ones(self.num_actions, dtype=np.float128) / num_actions
    self.t = 0
    self.beta = 2 * self.num_actions
    self.b = 0.2
  

class AIcrowdEvaluator:
  def __init__(self, kwargs):
    self.minx = None
    self.std = None
    self.unit = None
    self.render = False
    print(kwargs.items())
    for key, value in kwargs.items():
        o = f"{key}"
        a = f"{value}"
        if o == 'env_name':
          print("Reached env_name")
          self.env_name = a
        if o == 'num_players':
          self.num_players = int(a)
        if o == 'num_actions':
          self.num_actions = int(a)
        if o == 'num_iterations':
          self.num_iterations = int(a)
        if o == 'std':
          self.std = float(a)
        if o == 'unit':
          self.unit = float(a)
        if o == 'minx':
          self.minx = float(a)
        if o == 'agent_path':
          self.agent_path = a
        if o == 'render':
          self.render = bool(a)
    serializer = MessagePackSerializer()
    self.server = ZmqOracleServer(
      num_agent_connections=1,
      host=Constants.SERVER_HOST,
      port=Constants.SERVER_PORT,
      serializer=serializer,
    )
    if self.env_name == "DIR":
      self.env = DIR(num_actions=self.num_actions, std=self.std, num_players=2)
    if self.env_name == "SPA":
      self.env = SPA(num_actions=self.num_actions, std=self.std, num_players=self.num_players, unit=self.unit, minx=self.minx)
    else:
      self.env = Lemon(num_actions=self.num_actions, unit=self.unit, minx=self.minx, std=self.std, num_sellers=self.num_players-1)

  def serve(self):
    self.server.wait_for_agents()
    agents = []
    print(list(self.server.agents.values()))
    for i in range(self.env.num_players):
      agents.append(list(self.server.agents.values())[i])

    for t in range(self.num_iterations):
      actions = []
      for i in range(self.env.num_players):
        actions.append(agents[i].action_prob.astype(float))
      rewards, taken_actions = self.env.step(actions)

      for i in range(self.env.num_players):
        agents[i].feedback(taken_actions[i], rewards[i])
    
    self.server.close_agents()

  def evaluate(self):
    return {
        "score": compute_PoE(self.env) 
    }

if __name__ == "__main__":
  from threading import Thread
  from aicrowd_gym.clients.zmq_oracle_client import ZmqOracleClient
  import json

  env_name = "DIR"
  num_players = 2
  num_actions = 10
  num_iterations = 10000
  std = .1
  kwargs = {'env_name': env_name, 'num_players': num_players, 'num_actions': num_actions, 'num_iterations': num_iterations, 'std': std}
  print(kwargs)
  evaluator = AIcrowdEvaluator(kwargs=kwargs)

  server_thread = Thread(target=evaluator.serve, daemon=True)
  server_thread.start()

  oracle_client = ZmqOracleClient(
      host=Constants.SERVER_HOST,
      port=Constants.SERVER_PORT,
      serializer=MessagePackSerializer(),
  )


  for i in range(num_players):
    if env_name == "DIR":
      oracle_client.register_agent(EXP3DH(num_actions))
    if env_name == "Lemon":
      if i == 0:
        oracle_client.register_agent(EXP3DH(num_actions))
      else:
        oracle_client.register_agent(EXP3DH(2))
  
  oracle_client.run_agent()

  score = evaluator.evaluate()
  print(score)