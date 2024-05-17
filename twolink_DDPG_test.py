''' 
2024-04-25 ~ 2024-04-29
Updates: 
LagrangianDynamicsEnv
- Reward function: Changed to cosine similarity, originally incorrect euclidean distance
- Observation space: Changed angle [-inf, inf], originally [-pi, pi]
- self.state_coordinate: Corrected mistakes, originally link length was not multiplied
- render(): Added target rendering
- reset(): Erased render() for target rendering
- action space: Changed to continuous action with same maximum torque, originally discrete action space

twolink_DQN
- select_action: Corrected mistakes, originally [0.5 0.8, 0.2, 0.3] -> [1, 1] which is incorrect, now allocating action like this [[0, 0], [0, 1], [1, 0], [1, 1]]
- optimize_model: Corrected mistakes, originally action batch was [128, 2], now [128, 1]
- training loop: Changed method of action memory push, originally action itself i.e. [1, 1], now index of action i.e. [3]
- Codes modified to work for continuous action space
'''
'''
2024-04-30 ~
Updates:
LagrangianDynamicsEnv
- action space: Continuous action space

twolink_DDPN
- To cope with continuous action space, model changed to DDPG

reference: 'https://seunghan96.github.io/rl/38.(paper3)DDPG-%EC%BD%94%EB%93%9C%EB%A6%AC%EB%B7%B0/#1-import-packages'

2024-04-14 ~
Updates:
'''
 
import gymnasium as gym
import time
import matplotlib
import matplotlib.pyplot as plt
import argparse
from collections import deque, namedtuple
import numpy as np
import math
from drawnow import drawnow
import random
import csv

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init

from lagrangian_dynamics_env import LagrangianDynamicsEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with open('stateaction.csv', mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header if there is one
    points = [tuple(map(float, row)) for row in reader] # Convert each row to a tuple (or keep as list)

sa = torch.tensor(points, dtype=torch.float32, device=device)
state_de = sa[:,:4].clone()
action_de = sa[:,4:].clone()
next_state_de = state_de[1:].clone()
next_state_de = torch.cat((next_state_de, next_state_de[-1, :].unsqueeze(0)))
reward_de = torch.ones([146,1], dtype=torch.float32, device=device)
done_de = torch.zeros([146,1], dtype=torch.bool, device=device)
done_de[-1] = True

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))

last_score_plot = [0]
avg_score_plot = [0]



def draw_fig():
  plt.title('reward')
  plt.plot(last_score_plot, '-')
  plt.plot(avg_score_plot, 'r-')

def load_models_and_optimizers(actor, critic, actor_target, critic_target, actor_optimizer, critic_optimizer, filename):
    checkpoint = torch.load(filename)
    actor.load_state_dict(checkpoint['actor_state_dict'])
    critic.load_state_dict(checkpoint['critic_state_dict'])
    actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
    critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
    actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
    critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            
def visualize_weights(model, model_name):
    for name, param in model.named_parameters():
        if param.requires_grad:
            plt.figure()
            plt.title(f"{model_name} - {name}")
            plt.hist(param.detach().cpu().numpy().flatten(), bins=50)
            plt.show()
            
parser = argparse.ArgumentParser(description='PyTorch DDPG solution of TwoLinkRobotArm')

parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--tau', type=float, default=0.001)
parser.add_argument('--batch_size', type=float, default=64)
parser.add_argument('--max_episode', type=int, default=6000)
parser.add_argument('--max_explore_eps', type=int, default=4800)

cfg = parser.parse_args()

class Memory(object):
  def __init__(self, memory_size=100000):
    self.memory = deque([],maxlen=memory_size)
    self.memory_size = memory_size

  def __len__(self):
    return len(self.memory)

  def append(self, *args):
    self.memory.append(Transition(*args))

  def sample_batch(self, batch_size):
    return random.sample(self.memory, batch_size)


# Simple Ornstein-Uhlenbeck Noise generator
# Reference: https://github.com/rllab/rllab/blob/master/rllab/exploration_strategies/ou_strategy.py
def OUNoise():
  theta = 0.15
  sigma = 0.3
  mu = 0
  state = 0
  while True:
    yield state
    state += theta * (mu - state) + sigma * np.random.randn()


class Actor(nn.Module):
  def __init__(self):
    super(Actor, self).__init__()
    self.fc_1 = nn.Linear(4, 64)
    self.fc_2 = nn.Linear(64, 32)
    self.fc_out = nn.Linear(32, 2, bias=False)
    init.xavier_normal_(self.fc_1.weight)
    init.xavier_normal_(self.fc_2.weight)
    init.xavier_normal_(self.fc_out.weight)

  def forward(self, x):
    out = F.elu(self.fc_1(x))
    out = F.elu(self.fc_2(out))
    out = 10.0 * F.tanh(self.fc_out(out))
    return out


class Critic(nn.Module):
  def __init__(self):
    super(Critic, self).__init__()
    self.fc_state = nn.Linear(4, 32)
    self.fc_action = nn.Linear(2, 32)
    self.fc = nn.Linear(64, 128)
    self.fc_value = nn.Linear(128, 1, bias=False)
    init.xavier_normal_(self.fc_state.weight)
    init.xavier_normal_(self.fc_action.weight)
    init.xavier_normal_(self.fc.weight)
    init.xavier_normal_(self.fc_value.weight)

  def forward(self, state, action):
    out_s = F.elu(self.fc_state(state))
    out_a = F.elu(self.fc_action(action))
    out = torch.cat([out_s, out_a], dim=1)
    out = F.elu(self.fc(out))
    out = self.fc_value(out)
    return out


def get_action(_actor, state):
  if not isinstance(state, torch.Tensor):
    state = torch.from_numpy(state, dtype = torch.float32, device = device)
  action = _actor(state)
  action = torch.clamp(action, float(env.action_space.low[0]), float(env.action_space.high[0]))
  return action


def get_q_value(_critic, state, action):
  if not isinstance(state, torch.Tensor):
    state = torch.from_numpy(state, dtype = torch.float32, device = device)
  if not isinstance(action, torch.Tensor):
    action = torch.from_numpy(action, dtype = torch.float32, device = device)
  q_value = _critic(state, action)
  return q_value


def update_actor(state):
  action = actor(state)
  action = torch.clamp(action, float(env.action_space.low[0]), float(env.action_space.high[0]))
  # using chain rule to calculate the gradients of actor
  q_value = -torch.mean(critic(state, action))
  actor_optimizer.zero_grad()
  q_value.backward()
  actor_optimizer.step()
  return


def update_critic(state, action, target):
  q_value = critic(state, action)
  loss = F.mse_loss(q_value, target)
  critic_optimizer.zero_grad()
  loss.backward()
  critic_optimizer.step()
  return


def soft_update(target, source, tau):
  for target_param, param in zip(target.parameters(), source.parameters()):
    target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
env = LagrangianDynamicsEnv()

actor = Actor().to(device)
critic = Critic().to(device)
actor_target = Actor().to(device)
critic_target = Critic().to(device)
actor_target.load_state_dict(actor.state_dict())
critic_target.load_state_dict(critic.state_dict())

actor_optimizer = optim.Adam(actor.parameters(), lr=cfg.lr)
critic_optimizer = optim.Adam(critic.parameters(), lr=10*cfg.lr)

target = next_state_de
memory = Memory(memory_size=100000)
memory_de = Memory(memory_size=1000)
noise = OUNoise()
filename = '2000epsmodel.pth'
load_models_and_optimizers(actor, critic, actor_target, critic_target, actor_optimizer, critic_optimizer, filename)
print('weights loaded!')

for i in range(145):
  memory_de.append(state_de[i],action_de[i],reward_de[i],next_state_de[i],done_de[i])
def main():
  with torch.autograd.set_detect_anomaly(True):
    iteration_now = 0
    iteration = 0
    LAMBDA = 0.5
    state, _ = env.reset()
    state = torch.tensor(state-[np.pi/2,0,0,0], dtype=torch.float32, device=device)
    noise = OUNoise()
    episode = 0
    episode_score = 0
    episode_steps = 0
    # EPS_START = 0.9
    # EPS_END = 0.05
    # EPS_DECAY = 1000
    memory_warmup = cfg.batch_size * 3
    start_time = time.perf_counter()
    while episode < cfg.max_episode:
      print('\riter {}, ep {}'.format(iteration_now, episode), end='')
      # blend determinstic action with random action during exploration
      action = get_action(actor, state)
      # action = action_de[iteration_now]
      # blend determinstic action with random action during exploration
      observation, reward, terminated, truncated, _ = env.step(action.detach().cpu().numpy(), target[iteration_now].detach().cpu().numpy()-[np.pi/2,0,0,0])
      reward = torch.tensor([reward], dtype=torch.float32, device=device)
      done = terminated or truncated
      next_state = torch.tensor(observation-[np.pi/2,0,0,0], dtype=torch.float32, device=device)
      memory.append(state, action, reward, next_state, done)
      episode_score = episode_score + reward.item()
      episode_steps = episode_steps + 1
      iteration_now = iteration_now + 1
      iteration += 1
      if episode % 100 == 0:
        visualize_weights(actor, "Actor")
        visualize_weights(critic, "Critic")
      if done or iteration_now == 145:
        print(', score {:8f}, steps {}, ({:2f} sec/eps)'.
              format(episode_score, episode_steps, time.perf_counter() - start_time))
        avg_score_plot.append(avg_score_plot[-1] * 0.99 + episode_score * 0.01)
        last_score_plot.append(episode_score)
        drawnow(draw_fig)
        start_time = time.perf_counter()
        episode += 1
        episode_score = 0
        episode_steps = 0
        iteration_now = 0
        state, _ = env.reset()
        state = torch.tensor(state-[np.pi/2,0,0,0], dtype=torch.float32, device=device)
        noise = OUNoise()
      else:
        state = next_state.detach()
      
  env.close()


if __name__ == '__main__':
  main()
  plt.pause(0)
