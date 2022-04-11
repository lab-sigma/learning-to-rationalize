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

import os

from aicrowd_gym.servers.zmq_oracle_server import ZmqOracleServer
from aicrowd_gym.serializers import MessagePackSerializer
from loguru import logger
import logging

logging.disable('DEBUG')


class Constants:
    SERVER_HOST = os.getenv("AICROWD_REMOTE_SERVER_HOST", "0.0.0.0")
    SERVER_PORT = int(os.getenv("AICROWD_REMOTE_SERVER_PORT", "5000"))

  

class AIcrowdEvaluator:
  def __init__(self, kwargs):
    self.minx = None
    self.std = None
    self.unit = None
    self.render = False
    for key, value in kwargs.items():
        o = f"{key}"
        a = f"{value}"
        if o == 'env_name':
          self.env_name = a
        if o == 'num_players':
          self.num_players = int(a)
        if o == 'num_actions':
          self.num_actions = int(a)
        if o == 'num_iterations':
          print(f"Num_iterations: {num_iterations}")
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
    if self.env_name == "DIR":
      self.env = DIR(num_actions=self.num_actions, std=self.std, num_players=2)
    elif self.env_name == "SPA":
      self.env = SPA(num_actions=self.num_actions, std=self.std, num_players=self.num_players, unit=self.unit, minx=self.minx)
    else:
      self.env = Lemon(num_actions=self.num_actions, unit=self.unit, minx=self.minx, std=self.std, num_sellers=self.num_players-1)
    self.server = ZmqOracleServer(
      num_agent_connections=self.env.num_players,
      host=Constants.SERVER_HOST,
      port=Constants.SERVER_PORT,
      serializer=serializer,
    )

  def serve(self):
    self.server.wait_for_agents()
    agents = []
   
    for i in range(self.env.num_players):
      agents.append(list(self.server.agents.values())[i])

    for t in range(self.num_iterations):
      actions = []
      for i in range(self.env.num_players):
        request_id = agents[i].execute("compute_action")
        action = agents[i].get_response(request_id, timeout=1000).astype('float64')
        actions.append(action)
      
      rewards, taken_actions = self.env.step(actions)

      for i in range(self.env.num_players):
        agents[i].execute("feedback", taken_actions[i], rewards[i])
    
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
  evaluator = AIcrowdEvaluator(kwargs=kwargs)

  server_thread = Thread(target=evaluator.serve, daemon=True)
  server_thread.start()

  clients = []
  client_threads = []

  for i in range(num_players):
    oracle_client = ZmqOracleClient(
      host=Constants.SERVER_HOST,
      port=Constants.SERVER_PORT,
      serializer=MessagePackSerializer(),
    )
    clients.append(oracle_client)
    client_threads.append(Thread(target=oracle_client.run_agent, daemon=True))

  for i in range(num_players):
    if env_name == "DIR":
      clients[i].register_agent(EXP3DH(num_actions))
    if env_name == "Lemon":
      if i == 0:
        clients[i].register_agent(EXP3DH(num_actions))
      else:
        clients[i].register_agent(EXP3DH(2))
    
    client_threads[i].start()

  for client_thread in client_threads:
    client_thread.join()
    

  score = evaluator.evaluate()
