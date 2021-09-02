# AI Testing File

# Importing the libraries

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import torch.autograd as auto
from torch.autograd import Variable

# Building the AI architecture

class Network(nn.Module):
    
    def __init__(self, input_size, out_actions, hl_size=30):
        super(Network, self).__init__() # Gives the __init__ function access to the Network class and nn.module
        self.input_size = input_size # Save vars to class/self
        self.out_actions = out_actions
        self.hl_size = hl_size
        self.fc1 = nn.Linear(input_size, hl_size) # Input ---> Hidden Layer
        self.fc2 = nn.Linear(hl_size, out_actions) # Hidden Layer ---> Output
        
    def forward(self, state):
        x = state
        x = func.relu(self.fc1(x)) # Activate neurons and run x through
        qvals = self.fc2(x) # Run x through final layer
        return qvals # Return action probabilities
    
# Building the 'experience replay' module
        
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
            
    def samples(self, batch_size): # Try changing lambda to non-keyword
        samples = zip(*random.sample(self.memory, batch_size)) # if list = ((1,2, 3), (4, 5, 6)) --> zip(*list) = ((1, 4), (2, 3), (5, 6)) sorts (action reward state) into (action, action) (state, state) etc
        return map(lambda x: Variable(torch.cat(x, 0)), samples) # map applies function to all pieces of an iterable (like the list of samples). We use this to turn the list of samples into pytorch variables
    
# DEEP Q LEARNING BABY WOOOOOOOO LETS GO

class Dqn():
    
    def __init__(self, input_size, out_actions, gamma, memcap=100000, learning_rate=0.001, temp=7):
        self.gamma = gamma
        self.temp = temp
        self.reward_means = []
        self.model = Network(input_size, out_actions)
        self.memory = ReplayMemory(memcap)
        self.optimizer = optim.Adam(self.model.parameters(), learning_rate) # Try some different optimizers
        self.last_state = torch.Tensor(input_size).unsqueeze(0) # Trick network into thinking last state is in a batch, so it can properly process
        self.last_action = 0 # Placeholder value
        self.last_reward = 0 # Placeholder value
        
    def select_action(self, state):
        probs = func.softmax(self.model(Variable(state, volatile=True))*self.temp) # Save momory usage by ignoring gradient with volatile
        action = probs.multinomial(3) #Softmax amplifies difference between probabilities with increasing temperature
        return action.data[0, 0]

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1) # Unsqueeze to process, then squeeze (remove batch) to format output nicely
        next_outputs = self.model(batch_next_state).detach().max(1)[0] # Take the highest q-val action for each output
        target = self.gamma*next_outputs + batch_reward #Target is (gamma * potential reward) + current reward
        loss = func.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad() #Reset the optimizer
        loss.backward(retain_variables=True) # retain_variables also saves memory to speed up algorithim
        self.optimizer.step()
        
    def update(self, reward, signal):
        new_state = torch.Tensor(signal).float().unsqueeze(0) # Save current state as Tensor
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.LongTensor([self.last_reward]))) # Push current data to memory as tensors
        action = self.select_action(new_state) # Play the next action based on the new state
        if len(self.memory.memory) > 100: # Every step, get 100 state-action-reward-nextstate samples and learn from them
            batch_state, batch_next_state, batch_reward, batch_action = self.memory.samples(100)
            self.learn(batch_state, batch_next_state, torch.as_tensor(batch_reward, dtype=torch.float), batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_means.append(reward)
        if len(self.reward_means) > 1000: # Keep reward means within last 1000 rewards
            del self.reward_means.pop[0]        
        return action
        
    def score(self):
        return sum(self.reward_means)/(len(self.reward_means)+1)
        
    def save(self):
        torch.save({'state_dict':self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict,
                    }, 'last_brain.pth')
        
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("---> loading neural model...")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict']) # Load model with dictionary key
            self.optimizer.load_state_dict(checkpoint['optimizer']) # Load optimizer with dictionary key
            print("Neural model loaded.")
        else:
            print("No saved model found.")
        
        
        
        
        
        
        
