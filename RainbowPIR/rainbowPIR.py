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
from prioritized_replay import Prioritized
from IPython.display import clear_output
import matplotlib.pyplot as plt

#User chooses if he wants to use CUDA
if sys.argv[1]=='CUDA':
    USE_CUDA = torch.cuda.is_available()
else :
    USE_CUDA=False

Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

#Select environment
list_env = ["CartPole-v0", "LunarLander-v2", "Acrobot-v1"]
env_id = list_env[1]
env = gym.make(env_id)

#Noisy Networks
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
        
        
#Definition of the neural network
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
            nn.Linear(128, num_actions)
        )
        
        self.value = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.noisy1 = NoisyLinear(128,128)
        #TODO add some other noisy layers at the end of value & advantage ?

    #Passing a state through the network
    def forward(self, x):
        x = self.feature(x)
        x = self.noisy1(x)
        advantage = self.advantage(x)
        value = self.value(x)
        #Dueling DQN output
        return value + advantage - advantage.mean()
    
    #Choose best action
    def act(self, state):
        with torch.no_grad():
            state   = Variable(torch.FloatTensor(state).unsqueeze(0))
            q_value = self.forward(state)
            action  = q_value.max(1)[1].cpu().numpy()[0]
        return action

    def reset_noise(self):
        self.noisy1.reset_noise()
        
#Initialization of the 2 networks (Double DQN)
current_model = DQN(env.observation_space.shape[0], env.action_space.n)
target_model = DQN(env.observation_space.shape[0], env.action_space.n)
if USE_CUDA:
    current_model = current_model.cuda()
    target_model = target_model.cuda()
    
def update_target(current_model,target_model):
    target_model.load_state_dict(current_model.state_dict())

#Initialization of target network : identical to current network
update_target(current_model,target_model)

#Definition of the optimizer (for the loss minimization)
optimizer = optim.Adam(current_model.parameters())

#Initialization of Prioritized Experience Replay Buffer
replay_buffer = Prioritized(100000,0.5)


#Loss computation and network optimization
def compute_loss(batch_size, beta):
    
    #sample [batch_size] transitions
    nprev_state, action, rewards, nextstate, done, weights, idxes = replay_buffer.sample(batch_size, beta)
    nprev_state      = Variable(torch.FloatTensor(np.float32(nprev_state)))
    action     = Variable(torch.LongTensor(action))
    rewards     = Variable(torch.FloatTensor(np.float32(rewards)))
    rewards = torch.transpose(rewards,0,1) #size [n_multistep,batch_size]
    nextstate = Variable(torch.FloatTensor(np.float32(nextstate)))
    done       = Variable(torch.FloatTensor(done))
    done = torch.transpose(done,0,1) #size [n_multistep,batch_size]
    weights = Variable(torch.FloatTensor(weights))

    with torch.no_grad():
        donenew = abs(done - 1) #invert 0s and 1s in done 

    #Initialization of the multistep reward
    reward_n = Variable(torch.FloatTensor([0 for i in range(batch_size)]))

    #Computation of the multistep reward
    for i in range(n_multistep):
        if i>0:
            donenew[i] = donenew[i] * donenew[i-1] #donenew[i][j]=0 if one of the previous actions led to done episode   
        #reward_n is incremented only if the episode is not done 
        #(=> we don't take into account different episodes in the same multistep reward)
        #TODO check if really useful
        reward_n += donenew[i] * rewards[i]*gamma**(n_multistep-i-1)


    q_values      = current_model(nprev_state)
    with torch.no_grad():
        next_q_values = current_model(nextstate)
        next_q_state_values = target_model(nextstate)
        
    q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    #next q value = over the best action only
    next_q_value     = next_q_state_values.gather(1,torch.max(next_q_values,1)[1].unsqueeze(1)).squeeze(1)

    expected_q_value = reward_n + (gamma**n_multistep) * next_q_value * (1 - done[0])

    #Loss computation
    loss = (q_value - Variable(expected_q_value.data)).pow(2)
    loss = loss * weights
    #calculation of priorities prioritized replay 
    prios = loss**0.5 + 1e-5
    loss = loss.mean()
        
    #Optimization
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
    plt.savefig('rainbowPIR.png')
    plt.close()
    
    
num_frames = 1000000
num_episodes = 2000
batch_size = 64
gamma      = 0.99
n_multistep = 3

#Prioritized Replay parameter : linear increase for beta_frames frames, then constant
beta_start = 0.4
beta_frames = 1000 
beta_by_frame = lambda frame_idx: min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)

losses = []
all_rewards = []
episode_reward = 0
frame_idx = 0
render = False
state = env.reset()
states = [state for i in range(n_multistep+1)]
rewards = [0 for i in range(n_multistep)]
dones = [0 for i in range(n_multistep)]


while len(all_rewards) <= num_episodes :

    #Frame by frame update : prioritized replay parameter
    beta = beta_by_frame(frame_idx)
    
    #Take the same action for 4 frames
    if frame_idx % 4 == 0:
        action = current_model.act(states[0])
    
    next_state, reward, done, _ = env.step(action)
    
    #Multistep buffers rotation
    states = states[-1:] + states[:-1]
    states[0] = next_state
    rewards = rewards[-1:] + rewards[:-1]
    rewards[0] = reward
    dones = dones[-1:] + dones[:-1]
    dones[0] = done

    #Incrementation of prioritized replay buffer
    replay_buffer.push(states[n_multistep], action, rewards[:], states[0], dones[:])
    
    episode_reward += reward

    #Reset when finished episode
    if done:
        states[0] = env.reset()
        all_rewards.append(episode_reward)
        episode_reward = 0
        
    #Loss computation & network optimization
    if len(replay_buffer) > batch_size:
        loss = compute_loss(batch_size, beta)
        losses.append(loss.data)
        
    if frame_idx % 200 == 0:
        plot(frame_idx, all_rewards, losses)    
    
    if frame_idx % 100 == 0:
        update_target(current_model, target_model)
    
    #Rendering tool
    if sys.argv[2] == 'render':
        if frame_idx % 5000 == 0 :
            render = True    
        if render :
            if done :
                env.close()
                render = False
            else:
                env.render()
        
    if frame_idx % 1000 == 0:
        print("iteration",frame_idx)

    #Incrementation of the number of frames
    frame_idx += 1
