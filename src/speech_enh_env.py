# -*- coding: utf-8 -*-
"""
@author: Anurag Kumar
"""

import torch
import numpy as np
from utils import power_uncompress
from models.discriminator import batch_pesq
from collections import deque
import torch.nn.functional as F


class SpeechEnhancementAgent:
    def __init__(self, window, n_fft, hop, gpu_id=None):
        """
        State : Dict{noisy, clean, est_real, est_imag, cl_audio, est_audio}
        """
        self.gpu_id = gpu_id
        self.window = window
        self.n_fft = n_fft
        self.hop = hop
        #self.args = args
        #self.exp_buffer = replay_buffer(buffer_size, gpu_id=gpu_id)
        

    def set_batch(self, batch):
        self.state = batch
        self.clean = batch['clean']
        self.steps = batch['noisy'].shape[2]
        #self.noise = OUNoise(action_dim=batch['noisy'].shape[-1], gpu_id=self.gpu_id)

    def get_state_input(self, state, t):
        """
        Get the batched windowed input for time index t
        ARGS:
            t : time index

        Returns
            Batch of windowed input centered around t
            of shape (b, 2, f, w) 
        """
        state = state['noisy']
        b, _, tm, f = state.shape
        left = t - self.window
        right = t + self.window + 1
        if t < self.window: 
            pad = torch.zeros(b, 2, -left, f)
            if self.gpu_id is not None:
                pad = pad.to(self.gpu_id)
            windows = torch.cat([pad, state[:, :, :right, :]], dim=2)
        elif right > tm - 1:
            pad = torch.zeros(b, 2, right - tm, f)
            if self.gpu_id is not None:
                pad = pad.to(self.gpu_id)
            windows = torch.cat([state[:, :, left:, :], pad], dim=2) 
        else:
            windows = state[:, :, left:right, :]
        return windows 
    

class replay_buffer:
    def __init__(self, max_size, gpu_id=None):
        self.buffer = deque(maxlen=max_size)
        self.gpu_id = gpu_id

    def push(self, state, action, reward, next_state, t):
        experience = {'curr':state,
                      'action':action,
                      'reward':reward,
                      'next':next_state, 
                      't':t}
        self.buffer.append(experience)

    def sample(self):
        idx = np.random.choice(len(self.buffer), 1)[0]
        if self.gpu_id is None:
            retval = {'curr':{k:torch.FloatTensor(v) for k, v in self.buffer[idx]['curr'].items()},
                      'next':{k:torch.FloatTensor(v) for k, v in self.buffer[idx]['next'].items()},
                      'action':(torch.FloatTensor(self.buffer[idx]['action'][0]),
                                torch.FloatTensor(self.buffer[idx]['action'][1])),
                      'reward':torch.FloatTensor(self.buffer[idx]['reward']),
                      't':self.buffer[idx]['t']
                     }
        else:
            retval = {'curr':{k:torch.FloatTensor(v).to(self.gpu_id) for k, v in self.buffer[idx]['curr'].items()},
                      'next':{k:torch.FloatTensor(v).to(self.gpu_id) for k, v in self.buffer[idx]['next'].items()},
                      'action':(torch.FloatTensor(self.buffer[idx]['action'][0]).to(self.gpu_id),
                                torch.FloatTensor(self.buffer[idx]['action'][1]).to(self.gpu_id)),
                      'reward':torch.FloatTensor(self.buffer[idx]['reward']).to(self.gpu_id),
                      't':self.buffer[idx]['t']
                     }
        return retval

    def __len__(self):
        return len(self.buffer)

# Taken from #https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise(object):
    def __init__(self, action_dim, mu=0.0, theta=0.015, max_sigma=0.05, min_sigma=0.05, decay_period=100000, gpu_id=None):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_dim
        self.gpu_id = gpu_id
        self.reset()
        
        
    def reset(self):
        self.state = torch.ones(self.action_dim) * self.mu
        if self.gpu_id is not None:
            self.state = self.state.to(self.gpu_id)
        
    def evolve_state(self, action):
        x  = self.state
        action_dim = action[0].shape[-1]
        rand = torch.randn(action_dim)
        if self.gpu_id is not None:
            rand = rand.to(self.gpu_id)
        dx = self.theta * (self.mu - x) + self.sigma * rand
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0):
        ou_state = self.evolve_state(action)
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        mag_mask = torch.clip(action[0] + ou_state, torch.tensor(0.0).to(self.gpu_id), torch.max(action[0]))
        comp_mask = torch.clip(action[1] + ou_state.view(-1, 1), torch.min(action[1]), torch.max(action[1]))
        return (mag_mask, comp_mask)