import math, random

import sys
import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
from collections import deque
from replay_prio import Prioritized

from IPython.display import clear_output
import matplotlib.pyplot as plt

from common.wrappers import make_atari, wrap_deepmind, wrap_pytorch

if sys.argv[1]=='CUDA':
    USE_CUDA = torch.cuda.is_available()
else :
    USE_CUDA=False

Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)



class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, rewards, nextstate, done):
        #states      = np.expand_dims(states, 0)
#        print("\n\n PUSH ",rewards,"\n\n")
        self.buffer.append((state, action, rewards, nextstate, done))
#        print("\n\n BUFFER",self.buffer)
    
    def sample(self,batch_size):
        idxes = [random.randint(0, len(self) - 1) for _ in range(batch_size)]
        states, actions, rewards, nextstates, dones = [], [], [], [], []
        for i in idxes:
            data = self.buffer[i]
#            for j in range(len(data)-1):
#                print(data[j])
#                print('\n')
            state, action, reward, nextstate, done = data
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            nextstates.append(nextstate)
            dones.append(done)
#            print(np.array(rewards, copy=False))
#            print("\n Data ")
        return np.array(states, copy=False), np.array(actions, copy=False), np.array(rewards, copy=False), np.array(nextstates, copy=False), np.array(dones, copy=False)


#    def sample(self, batch_size):
#        action, rewards, states, done = zip(*random.sample(self._buffer, batch_size))
#        return np.array(action), np.array(rewards), np.array(states), np.array(done)
    
    def __len__(self):
        return len(self.buffer)
        
        
        
env_id = "CartPole-v0"
#env_id = "LunarLander-v2"
#env_id = "Acrobot-v1"
env = gym.make(env_id)

epsilon_start = 1.0
epsilon_final = 0.05
epsilon_decay = 500

epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

## Epsilon-greedy :
epsilon = epsilon_start
epsilon_decay_length = 100000
epsilon_decay_exp = 0.97
epsilon_linear_step = (epsilon_start-epsilon_final)/epsilon_decay_length


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        
        self.in_features  = in_features
        self.out_features = out_features
        self.std_init     = std_init
        
        self.weight_mu    = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        
        self.bias_mu    = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def forward(self, x):
        if self.training: 
            weight = self.weight_mu + self.weight_sigma.mul(Variable(self.weight_epsilon))
            bias   = self.bias_mu   + self.bias_sigma.mul(Variable(self.bias_epsilon))
        else:
            weight = self.weight_mu
            bias   = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))
        
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))
    
    def reset_noise(self):
        epsilon_in  = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))
    
    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x
        



class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DQN, self).__init__()
        
        self.feature = nn.Sequential(
            nn.Linear(num_inputs,128),
            nn.ReLU()
        )
        
        self.advantage = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions) #Noisy --> NO
        )
        
        self.value = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.noisy1 = NoisyLinear(128,128)
    #    self.noisy2 = NoisyLinear(128,env.action_space.n)
   #     self.noisy3 = NoisyLinear(128,1)
        
    def forward(self, x):
        x = self.feature(x)
        x = self.noisy1(x)
        advantage = self.advantage(x)
  #      advantage = self.noisy2(advantage)
        value = self.value(x)
 #       value = self.noisy3(value)
        return value + advantage - advantage.mean()
    
    def act(self, state, epsilon):
        #if random.random() > epsilon:
        if 1==1:
            with torch.no_grad():
                state   = Variable(torch.FloatTensor(state).unsqueeze(0))
                q_value = self.forward(state)
                action  = q_value.max(1)[1].cpu().numpy()[0]
        else:
            action = random.randrange(env.action_space.n)
        return action
        
    def reset_noise(self):
        self.noisy1.reset_noise()
#        self.noisy2.reset_noise()
#        self.noisy3.reset_noise()
        
        
        
current_model = DQN(env.observation_space.shape[0], env.action_space.n)
target_model = DQN(env.observation_space.shape[0], env.action_space.n)

if USE_CUDA:
    current_model = current_model.cuda()
    target_model = target_model.cuda()

def update_target(current_model,target_model):
    target_model.load_state_dict(current_model.state_dict())

update_target(current_model,target_model)


optimizer = optim.Adam(current_model.parameters())
#, lr = 0.00025
replay_buffer = Prioritized(100000,0.5)
#replay_buffer = ReplayBuffer(100000)

def compute_td_loss(batch_size, beta):
    
    state, action, rewards, nextstate, done, weights, idxes = replay_buffer.sample(batch_size, beta)
    #state, action, rewards, nextstate, done = replay_buffer.sample(batch_size)

    state      = Variable(torch.FloatTensor(np.float32(state)))
    action     = Variable(torch.LongTensor(action))
    rewards     = Variable(torch.FloatTensor(np.float32(rewards)))
    rewards = torch.transpose(rewards,0,1)
    nextstate = Variable(torch.FloatTensor(np.float32(nextstate)))
    done       = Variable(torch.FloatTensor(done))
    done = torch.transpose(done,0,1)
    weights = Variable(torch.FloatTensor(weights))

    with torch.no_grad():
        donenew = abs(done - 1) 
#        donenew = done  
    
#    reward_n = 0
    reward_n = Variable(torch.FloatTensor([0 for i in range(done.size(1))]))
#    isdone = [False for j in range(done.size(1))]
    for i in range(n):
#        for j in range(done.size(1)):
#            if done[i][j]==1:
#                isdone[j] = True
#            if not isdone[j]:
#                reward_n[j] += rewards[i][j]*gamma**(n-i-1)
        if i>0:
            donenew[i] = donenew[i] * donenew[i-1] 
        reward_n += donenew[i] * rewards[i]*gamma**(n-i-1)

    q_values      = current_model(state)
    with torch.no_grad():
        next_q_values = current_model(nextstate)
        next_q_state_values = target_model(nextstate)

    q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value     = next_q_state_values.gather(1,torch.max(next_q_values,1)[1].unsqueeze(1)).squeeze(1)

    expected_q_value = reward_n + (gamma**n) * next_q_value * (1 - done[0])

    loss = (q_value - Variable(expected_q_value.data)).pow(2)
    loss = loss * weights
    prios = loss**0.5 + 1e-5
    loss = loss.mean()
        
    optimizer.zero_grad()
    loss.backward()
    replay_buffer.update_priorities(idxes,prios.data.cpu().numpy())
    optimizer.step() 
    
    current_model.reset_noise()
    target_model.reset_noise()
    
    return loss
    
    
def plot(frame_idx, rewards, losses):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.savefig('rainbow_empty9_05cart_OK.png')
    plt.close()
    
    
num_frames = 1000000
num_episodes = 2000
batch_size = 64
gamma      = 0.99
n = 3 #Multistep

#replayInitial = 10000

beta_start = 0.4
beta_frames = 1000 
beta_by_frame = lambda frame_idx: min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)

losses = []
all_rewards = []
episode_reward = 0
frame_idx = 0
render = False
state = env.reset()
states = [state for i in range(n+1)] #Multistep
#states = []
#rewards = torch.Tensor([0 for i in range(n)]) #Multistep
rewards = [0 for i in range(n)]
dones = [0 for i in range(n)]

#for frame_idx in range(1, num_frames + 1):

while len(all_rewards) <= num_episodes :

    beta = beta_by_frame(frame_idx)
    epsilon = epsilon_by_frame(frame_idx)
    
    if frame_idx % 4 == 0:
        action = current_model.act(states[0],epsilon)
    
#    action = current_model.act(states[0], epsilon)
    next_state, reward, done, _ = env.step(action)
    
    #Rotate (scalars)
    states = states[-1:] + states[:-1]
    states[0] = next_state
    
    for i in range(n-1,0,-1):
        rewards[i] = rewards[i-1]
    rewards[0] = reward
    for i in range(n-1,0,-1):
        dones[i] = dones[i-1]
    dones[0] = done
    
    replay_buffer.push(states[n], action, rewards[:], states[0], dones[:])
    
#    state = next_state
    episode_reward += reward

#    if frame_idx < epsilon_decay_length:
#        epsilon -= epsilon_linear_step
#    elif done:    
#        epsilon *= epsilon_decay_exp

    if done:
        states[0] = env.reset()
        all_rewards.append(episode_reward)
        episode_reward = 0
        
    if len(replay_buffer) > batch_size:
        loss = compute_td_loss(batch_size, beta)
        losses.append(loss.data)
        
    if frame_idx % 200 == 0:
        plot(frame_idx, all_rewards, losses)    
    
    if frame_idx % 100 == 0:
        update_target(current_model, target_model)
    
    if sys.argv[2] == 'render':
        if frame_idx % 5000 == 0 :
            render = True    
        if render :
            if done :
                env.close()
                render = False
            else:
                env.render()
        
    
    frame_idx += 1

    if frame_idx % 1000 == 0:
        print("iteration",frame_idx)
