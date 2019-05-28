# -*- coding: utf-8 -*-
"""
Created on Mon May 20 14:26:46 2019

@author: paulc
"""

import numpy as np
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

from common.layers import NoisyLinear
from common.replay_buffer_prio import PrioritizedReplayBuffer
from common.wrappers import make_atari, wrap_deepmind, wrap_pytorch

#from IPython.display import clear_output
import matplotlib.pyplot as plt

#USE_CUDA = False
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)


def save_network(target_network,env_id):
    print("Saving model...")
    if not os.path.exists('./networks/'):
        os.mkdir('./networks/')
    torch.save(target_network.state_dict(), './networks/' + env_id + '_trained_network.pt' )
    print("Model saved !")


#Initialize target (= current model)
def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())


#C51 Projection algorithm
def projection_distribution(next_state, rewards, dones):
    batch_size = next_state.size(0)

    #next_state : dim ( batch_size, 1,
    #rewards : dim (batch_size)
    #dones : dim (batch_size)
    #support : dim (num_atoms)

    delta_z = float(Vmax - Vmin) / (num_atoms - 1)

    #support = z (what we want to project onto)
    support = torch.linspace(Vmin, Vmax, num_atoms)

    #target_model(next_state) <=> forward(next_state) = distribution p(x_t+1,a) over actions
    #dim (batch_size,num_actions,num_atoms)
    next_dist   =  target_model(next_state).data.cpu() * support

    #sum over the atoms & search index of the best action
    #(a* = argmaxQ(x_t+1,a))
    #dim (batch_size)
    next_action = next_dist.sum(2).max(1)[1]

    #dim (batch_size,1,num_atoms)
    next_action = next_action.unsqueeze(1).unsqueeze(1).expand(next_dist.size(0), 1, next_dist.size(2))

    #gather : reorganizes elements
    #distribution over the best action only
    #dim (batch_size,num_atoms)
    next_dist   = next_dist.gather(1, next_action).squeeze(1)

    #change dimensions
    #rewards : dim (batch_size, num_atoms)
    #dones : dim (batch_size, num_atoms)
    #support : dim (batch_size, num_atoms)
    rewards = rewards.unsqueeze(1)
    rewards = rewards.expand_as(next_dist)
    dones   = dones.unsqueeze(1).expand_as(next_dist)
    support = support.unsqueeze(0).expand_as(next_dist)

    #TODO gamma**3 ??
    Tz = rewards + (1 - dones) * gamma**3 * support
#    Tz = rewards + (1 - dones) * 0.99 * support
    Tz = Tz.clamp(min=Vmin, max=Vmax)
    b  = (Tz - Vmin) / delta_z
    l  = b.floor().long()
    u  = b.ceil().long()

    #batch_size rows, num_atoms columns, each row = i*num_atoms repeated on the whole row
    offset = torch.linspace(0, (batch_size - 1) * num_atoms, batch_size).long()\
                    .unsqueeze(1).expand(batch_size, num_atoms)

    proj_dist = torch.zeros(next_dist.size())

    proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
    proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

    #proj_dist : dim (batch_size, num_atoms)
    return proj_dist


def compute_KL_loss(batch_size):

    #replay batch sample :
    action, rewards, states, done, _, indices = replay_buffer.sample(batch_size,beta)

    #transform batch sample into floats
    states = Variable(torch.FloatTensor(np.float32(states)), requires_grad=False)
    states = states.permute(1,0,2,3,4)
    action     = Variable(torch.LongTensor(action))
    rewards     = torch.FloatTensor(rewards)
    rewards = torch.transpose(rewards,0,1)
    done       = torch.FloatTensor(np.float32(done))

    reward_n = rewards[0] * gamma**2 + rewards[1] * gamma + rewards[2]

    #projection on the distribution space (C51)
    #dim (batch_size,num_atoms)
    proj_dist = projection_distribution(states[0], reward_n, done).to(device)

    #forward(state)
    dist = current_model(states[3]).to(device)

    action = action.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, num_atoms)
    dist = dist.gather(1, action).squeeze(1)
    dist.data.clamp_(0.01, 0.99)

    loss = F.kl_div(proj_dist,dist)

    priorities = F.kl_div(proj_dist,dist,reduction='none')
    priorities = [abs(priorities[i].mean().data.cpu().numpy())**omega for i in range(batch_size)]

    optimizer.zero_grad()
    loss.backward()
    replay_buffer.update_priorities(indices,priorities)

    optimizer.step()
    current_model.reset_noise()
    target_model.reset_noise()


    return loss






def plot(frame_idx, rewards, losses,env_id):
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.savefig(env_id+"results.png")

#############################

list_id=["PongNoFrameskip-v4","BreakoutNoFrameskip-v4"]

#env_id = list_id[int(sys.argv[0])]

env_id = list_id[0]

print("Playing on Game "+env_id)
env    = make_atari(env_id)
env    = wrap_deepmind(env)
env    = wrap_pytorch(env)
############################


class RainbowCnnDQN(nn.Module):

    def __init__(self, input_shape, num_actions, num_atoms, Vmin, Vmax):
        super(RainbowCnnDQN, self).__init__()

        self.input_shape   = input_shape
        self.num_actions  = num_actions
        self.num_atoms    = num_atoms
        self.Vmin         = Vmin
        self.Vmax         = Vmax

        #Definition of the neural network
        #Conv2d(input_shape, output_shape, kernel_size, stride)
        #(kernel_size = size of the convolutional filter)
        #(stride = step used to shift the convolutional filter)
        #ReLu --> max(0,x)
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        #NoisyLinear( in_features, out_features, use_cuda)
        #(Layers after the features)
        self.noisy_value1 = NoisyLinear(self.feature_size(), 512, use_cuda=USE_CUDA)
        self.noisy_value2 = NoisyLinear(512, self.num_atoms, use_cuda=USE_CUDA)

        self.noisy_advantage1 = NoisyLinear(self.feature_size(), 512, use_cuda=USE_CUDA)
        self.noisy_advantage2 = NoisyLinear(512, self.num_atoms * self.num_actions, use_cuda=USE_CUDA)


    # passes a state through the neural network, gives distributional output (1,actions,atoms)
    def forward(self, x):

        batch_size = x.size(0)

        #Colored pixels
        x = x / 255.
        #x goes through the NN
        x = self.features(x)
        #Reshape x with batch_size rows & adapted number of columns
        x = x.view(batch_size, -1)

        #Output num_atoms features lists
        value = F.relu(self.noisy_value1(x))
        value = self.noisy_value2(value)

        #Output num_atoms * num_actions features lists
        advantage = F.relu(self.noisy_advantage1(x))
        advantage = self.noisy_advantage2(advantage)

        #Reshape value & advantage
        value     = value.view(batch_size, 1, self.num_atoms)
        advantage = advantage.view(batch_size, self.num_actions, self.num_atoms)

        #Factorization of action values : DUELING NETWORKS
        #(mean only over the different actions --> over 2d dimension of advantage)
        # here x = q(s,a)
        x = value + advantage - advantage.mean(1, keepdim=True)

        #DISTRIBUTIONAL RL
        # softmax => for each action, returns num_atoms probabilities
        x = F.softmax(x.view(-1, self.num_atoms),dim=1).view(-1, self.num_actions, self.num_atoms)

        #dim : ( 1, num_actions, num_atoms)
        return x


    def reset_noise(self):
        self.noisy_value1.reset_noise()
        self.noisy_value2.reset_noise()
        self.noisy_advantage1.reset_noise()
        self.noisy_advantage2.reset_noise()

    #Size of the output of the features (before passing through noisynets)
    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

    #returns the index of the best action to choose
    def act(self, state):
        #unsqueeze(0) = adds a "1" dimension at index 0 of shape ; make it float.
        #Volatile : doesn't need much memory because we won't do any backprop
        state = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), volatile=True)

        #.cpu() moves the tensor to the cpu
        #.data : content of the tensor
        dist = self.forward(state).data.cpu()

        #torch.linspace : 1D tensor length num_atoms, equally spaced points in [Vmin,Vmax]
        #dist : shape = ( ?, num_actions, num_atoms)
        # * => multiplies each (1,num_actions) matrix by a number in [Vmin,Vmax]
        dist = dist * torch.linspace(self.Vmin, self.Vmax, self.num_atoms)

        #dist.sum(2) = sum over dimension 2 => shape(?,num_actions)
        # => sum the results of all the atoms
        # .max(1) =>  maximizes on the different actions
        # [1] => indices of the max*
        # .numpy() => transforms into an array
        # [0] => index of the action to choose
        action = dist.sum(2).max(1)[1].numpy()[0]
        return action


num_atoms = 51
Vmin = -10
Vmax = 10

#10000
replay_initial = 1000
num_frames = 1000000
batch_size = 32

alpha = 0.00025/4
beta = 0.5
gamma = 0.99
omega = 0.5
n = 3


#Define models
current_model = RainbowCnnDQN(env.observation_space.shape, env.action_space.n, num_atoms, Vmin, Vmax)
target_model  = RainbowCnnDQN(env.observation_space.shape, env.action_space.n, num_atoms, Vmin, Vmax)

#Transfer to cuda
if USE_CUDA:
    current_model = current_model.cuda()
    target_model  = target_model.cuda()

#lr = learning_rate
optimizer = optim.Adam(current_model.parameters(), lr=alpha)

#target_model = current_model
update_target(current_model, target_model)

#list of (state, action, reward, next_state, done) elements
#0.5 => importance of prioritization = 1/2
replay_buffer  = PrioritizedReplayBuffer(100000,0.5)

#Initializations
all_rewards = []
rewards = torch.tensor([0,0,0])
episode_reward = 0
losses = []
loss=torch.tensor(0)

#initialize state
state = env.reset()
states = [state,state,state,state]


for frame_idx in range(1, num_frames + 1):

    #linear increase of beta
    if frame_idx<80000:
        beta += 0.6/80000

    # choose best action corresponding to state
    action = current_model.act(state)

    #make transition (environment only)
    next_state, reward, done, _ = env.step(action)


    #Rotate
    states = states[-1:] + states[:-1]
    states[0] = next_state

    rewards[2] = rewards[1]
    rewards[1] = rewards[0]
    rewards[0] = reward
    #update the replay buffer with current data
    replay_buffer.push(action , rewards , states , done)

    state = next_state

    episode_reward += reward

    #finished episode
    if done:
        #begin new episode => initialize state
        state = env.reset()
        #all_rewards = list of the final reward for each episodes
        all_rewards.append(episode_reward)
        episode_reward = 0

    # first iterations = not interesting to compute loss
    if len(replay_buffer) > replay_initial:
        loss = compute_KL_loss(batch_size)
        losses.append(loss.data.cpu().numpy())

    if frame_idx % 100 == 0:
        print(frame_idx)
        if loss.data.cpu().numpy()!=0:
            print("loss : ",loss.data.cpu().numpy())


    #save network in files & plot
    if frame_idx % 10000 == 0:
        save_network(target_model,env_id)
        plot(frame_idx, all_rewards, losses,env_id)

    #Only update target network each 1000 iterations
    if frame_idx % 1000 == 0:
        update_target(current_model, target_model)


# =============================================================================
# TODO
#   - No DDQN
#   - No Prioritized Replay, only Experience Replay (easy to change)
#   - Dueling Networks OK (in forward())
#   - Multi-step : NO
#   - C51 : OK (Simplified ?)
#   - Noisy Nets : OK (add factorised gaussian noise ?)

#	- Why (1-dones) ?

#   Utiliser la loss D_KL + prioritize avec la norme KL !
# =============================================================================
