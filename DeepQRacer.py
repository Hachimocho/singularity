#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 12:16:56 2019

@author: Bryce Gernon
"""

import gym
import numpy
import torch
import torch.nn as nn
import torch.nn.Functional as F
import random
env = gym.make('Car-Racing-v0')
env.reset()
steps = 1000
cs = env.action_space
os = env.observation_space

print(cs)
print(os)

for x in steps:
    env.render()
    env.step(cs.sample())