
import random
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from .actor_critic import Actor, Critic
from .memory import SequentialMemory
from lib.utils.random_process import TruncatedNormalProcess
from lib.utils.utils import to_numpy, to_tensor

# from ipdb import set_trace as debug

criterion = nn.MSELoss()
USE_CUDA = torch.cuda.is_available()

class DDPG(object):
    def __init__(self, nb_states, nb_actions, num_layers, args):
        
        if args.seed > 0:
            self.seed(args.seed)

        self.nb_states = nb_states
        self.nb_actions= nb_actions
        
        # Create Actor and Critic Network
        net_cfg = {
            'hidden1':args.hidden1, 
            'hidden2':args.hidden2, 
            'init_w':args.init_w
        }
        self.actor = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_target = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_optim  = Adam(self.actor.parameters(), lr=args.lr_a)

        self.critic = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_target = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_optim  = Adam(self.critic.parameters(), lr=args.lr_a)

        self.hard_update(self.actor_target, self.actor) # Make sure target is with the same weight
        self.hard_update(self.critic_target, self.critic)
        
        #Create replay buffer
        self.memory = SequentialMemory(limit=args.rmsize, window_length=args.window_length)
        self.random_process = TruncatedNormalProcess(sigma=args.init_delta, decay_rate=args.delta_decay)
        self.random_process.reset_states()

        # Hyper-parameters
        self.batch_size = args.bsize
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 0.0004
        self.lbound = 0.0
        self.rbound = 1.0

        # 
        self.epsilon = 0.3
        self.epsilon_bound = 0.5 / (self.nb_actions*num_layers)
        # self.s_t = None # Most recent state
        # self.a_t = None # Most recent action
        self.is_training = True

        # 
        if USE_CUDA: self.cuda()

        # moving average baseline
        self.moving_average = None
        self.moving_alpha = 0.5  # based on batch, so small

    def update_policy(self):
        # Sample batch
        state_batch, action_batch, reward_batch, \
        next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)

        # normalize the reward
        batch_mean_reward = np.mean(reward_batch)
        if self.moving_average is None:
            self.moving_average = batch_mean_reward
        else:
            self.moving_average += self.moving_alpha * (batch_mean_reward - self.moving_average)
        reward_batch -= self.moving_average

        # Prepare for the target q batch
        with torch.no_grad():
            next_q_values = self.critic_target([
                to_tensor(next_state_batch),
                self.actor_target(to_tensor(next_state_batch)),
            ])

        target_q_batch = to_tensor(reward_batch) + \
            self.discount*to_tensor(terminal_batch.astype(np.float64))*next_q_values

        # Critic update
        self.critic.zero_grad()

        q_batch = self.critic([ to_tensor(state_batch), to_tensor(action_batch) ])
        
        value_loss = criterion(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()

        policy_loss = -self.critic([
            to_tensor(state_batch),
            self.actor(to_tensor(state_batch))
        ])

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        # Target update
        self.soft_update(self.actor_target, self.actor, self.tau)
        self.soft_update(self.critic_target, self.critic, self.tau)

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def observe(self, r_t, s_t, a_t, done):
        if self.is_training:
            self.memory.append(s_t, a_t, r_t, done)
            # self.s_t = s_t1
    
    def baseline_action(self, mask_hint):
        action = np.random.uniform(self.lbound, self.rbound, self.nb_actions)
        mask = np.asarray(mask_hint)
        action = action / 2 + self.rbound * mask / 2 + self.lbound * (1-mask) /2        
        return action

    def random_action(self):
        action = np.random.uniform(self.lbound, self.rbound, self.nb_actions)
        # self.a_t = action
        return action

    def select_action(self, s_t, decay_epsilon=True):
        # if random.random() < self.epsilon:
        #     action = self.random_action()
        # else:
        action = to_numpy(
            self.actor(to_tensor(np.array([s_t])))
        ).squeeze(0)
        # action += self.is_training*max(self.epsilon, 0)*self.random_process.sample()
        # action = self.random_process(action)
        if self.is_training:
            for i in range(self.nb_actions):
                action[i] = self.random_process.sample(action[i])
                # if random.random() < self.epsilon:
                #     action[i] = np.random.uniform(self.lbound, self.rbound)
        action = np.clip(action, self.lbound, self.rbound)

        # self.a_t = action
        # print(action)
        return action

    def reset(self, obs):
        # self.s_t = obs
        # self.random_process.reset_states()
        pass

    def step(self, reset=False):
        self.random_process.step(reset)
        if self.epsilon > self.epsilon_bound:
            self.epsilon -= self.depsilon
        if reset:
            self.epsilon = 0.05

    def load_weights(self, output):
        if output is None: return

        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(output))
        )

        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(output))
        )


    def save_model(self,output):
        torch.save(
            self.actor.state_dict(),
            '{}/actor.pkl'.format(output)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/critic.pkl'.format(output)
        )

    def seed(self,s):
        torch.manual_seed(s)
        if USE_CUDA:
            torch.cuda.manual_seed(s)

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )
    
    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(param.data)
