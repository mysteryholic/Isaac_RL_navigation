import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import math

class CBAM(nn.Module):
    def __init__(self, channels, r):
        super(CBAM, self).__init__()
        self.channels = channels
        self.r = r
        self.sam = SAM(bias=False)
        self.cam = CAM(self.channels, r=self.r)

    def forward(self, x):
        output = self.cam(x)
        output = self.sam(output)
        return output
    
class SAM(nn.Module):
    def __init__(self, bias=False):
        super(SAM, self).__init__()
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, bias=self.bias)

    def forward(self, x):
        max = torch.max(x,1)[0].unsqueeze(1)
        avg = torch.mean(x,1).unsqueeze(1)
        concat = torch.cat((max,avg), dim=1)
        output = self.conv(concat)
        output = output * x
        return output

class CAM(nn.Module):
    def __init__(self, channels, r):
        super(CAM, self).__init__()
        self.channels = channels
        self.r = r
        self.linear_max = nn.Sequential(
            nn.Linear(in_features = self.channels, out_features = self.channels//self.r, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features = self.channels//self.r, out_features = self.channels, bias=True))

    def forward(self, x):
        max = F.adaptive_max_pool2d(x, output_size=1)
        avg = F.adaptive_avg_pool2d(x, output_size=1)
        b, c, _, _ = x.size()
        linear_max = self.linear_max(max.view(b,c)).view(b, c, 1, 1)
        linear_avg = self.linear_max(avg.view(b,c)).view(b, c, 1, 1)
        output = linear_max + linear_avg
        output = F.sigmoid(output) * x
        return output
    
class Actor(nn.Module):
    def __init__(self,input_shape,state_dim, action_dim,hidden_dim):
        super(Actor, self).__init__()

        self.layer_1 = nn.Linear(state_dim + np.prod(input_shape), hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s,s_a):
        b, _, _, _ = s.size()
        s = s.view(b,-1)
        s = torch.concat((s,s_a), 1)
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a

class Critic(nn.Module):
    def __init__(self, input_shape, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        self.layer_1 = nn.Linear(state_dim + np.prod(input_shape), hidden_dim)
        self.layer_2_s = nn.Linear(hidden_dim, hidden_dim)
        self.layer_2_a = nn.Linear(action_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, action_dim)

        self.layer_4 = nn.Linear(state_dim + np.prod(input_shape), hidden_dim)
        self.layer_5_s = nn.Linear(hidden_dim, hidden_dim)
        self.layer_5_a = nn.Linear(action_dim, hidden_dim)
        self.layer_6 = nn.Linear(hidden_dim, action_dim)

    def forward(self, s, s_a, a):
        b, _, _, _ = s.size()
        s = s.view(b,-1)
        s = torch.concat((s,s_a), 1)

        s1 = F.relu(self.layer_1(s))
        self.layer_2_s(s1)
        self.layer_2_a(a)
        s11 = torch.mm(s1, self.layer_2_s.weight.data.t())
        s12 = torch.mm(a, self.layer_2_a.weight.data.t())
        s1 = F.relu(s11 + s12 + self.layer_2_a.bias.data)
        q1 = self.layer_3(s1)

        s2 = F.relu(self.layer_4(s))
        self.layer_5_s(s2)
        self.layer_5_a(a)
        s21 = torch.mm(s2, self.layer_5_s.weight.data.t())
        s22 = torch.mm(a, self.layer_5_a.weight.data.t())
        s2 = F.relu(s21 + s22 + self.layer_5_a.bias.data)
        q2 = self.layer_6(s2)

        return q1,q2

class TD3(object):
    def __init__(self, input_shape, num_inputs_add, action_space, args):
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.hidden_size = args.hidden_size
        self.policy_freq = args.policy_freq

        self.policy_type = args.policy
        self.count = 0
        state_shape=input_shape[0]*input_shape[1]*input_shape[2]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actor = Actor(state_shape, num_inputs_add, action_space, self.hidden_size).to(self.device)
        self.actor_target = Actor(state_shape, num_inputs_add, action_space, self.hidden_size).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(state_shape, num_inputs_add, action_space, self.hidden_size).to(self.device)
        self.critic_target = Critic(state_shape, num_inputs_add, action_space, self.hidden_size).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
        
        self.cbam= CBAM(channels=input_shape[0], r=3).to(self.device)
        self.cbam_optimizer = torch.optim.Adam(self.cbam.parameters(), lr=0.001)
        

    def select_action(self, state, state_add):
        print("select_action using")
        state = torch.FloatTensor(state).to(self.device)
        state = state.unsqueeze(0)
        state_add = torch.FloatTensor(state_add).to(self.device)
        state_add = state_add.unsqueeze(0)
        state = self.cbam(state)
        return self.actor(state,state_add).cpu().data.numpy().flatten()
    
    def update_parameters(self, memory, args):
        max_Q=float('-inf')
        av_Q=0
        av_loss=0
        state_batch, state_batch_add, action_batch, reward_batch, next_state_batch, next_state_batch_add, mask_batch = memory.sample(batch_size=args.batch_size)

        state_batch1 = torch.FloatTensor(state_batch).to(self.device)
        state_batch_add = torch.FloatTensor(state_batch_add).to(self.device)
        next_state_batch1 = torch.FloatTensor(next_state_batch).to(self.device)
        next_state_batch_add = torch.FloatTensor(next_state_batch_add).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device).view(-1, 1)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).view(-1, 1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).view(-1, 1)

        state_batch = self.cbam(state_batch1)
        next_state_batch = self.cbam(next_state_batch1)

        next_action = self.actor_target(next_state_batch, next_state_batch_add)

        noise = torch.randn_like(action_batch).to(self.device)
        noise = noise.clamp(-0.2, 0.2)
        next_action = (next_action + noise).clamp(-math.pi/4,math.pi/4)

        target_Q1, target_Q2 = self.critic_target(next_state_batch, next_state_batch_add, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        av_Q += torch.mean(target_Q)
        max_Q = max(max_Q, torch.max(target_Q))
        target_Q = reward_batch + ((1 - mask_batch)* self.gamma * target_Q).detach()
        
        current_Q1, current_Q2 = self.critic(state_batch, state_batch_add, action_batch)
        loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.critic_optimizer.step()

        if self.count % self.policy_freq == 0:
            actor_grad, _ = self.critic(state_batch, state_batch_add, self.actor(state_batch, state_batch_add))
            actor_grad = -actor_grad.mean()
            self.actor_optimizer.zero_grad()
            actor_grad.backward()
            self.actor_optimizer.step()
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        av_loss += loss
        av_loss /= (self.count + 1)
        self.count += 1
        self.update_cbam(next_state_batch1,next_state_batch_add, next_action, target_Q)
        return av_loss, av_Q, max_Q
    
    def update_cbam(self, state,state_add,action, target_Q):
        current_cbam= self.cbam(state)
        current_Q1,_=self.critic(current_cbam, state_add, action)
        loss=(target_Q-current_Q1).mean()
        self.cbam_optimizer.zero_grad()
        loss.backward()
        self.cbam_optimizer.step()
        

    # Save model parameters
    def save_checkpoint(self, directory, filename, i):
        torch.save(self.actor.state_dict(), "%s/%s/%s_actor_%d.pth" % (directory, filename, filename, i))
        torch.save(self.critic.state_dict(), "%s/%s/%s_critic_%d.pth" % (directory, filename, filename, i))
    
    # Load model parameters
    def load_checkpoint(self, directory, filename):
        self.actor.load_state_dict(torch.load("%s/%s/%s_actor.pth" % (directory, filename, filename), weights_only=True))
        self.critic.load_state_dict(torch.load("%s/%s/%s_critic.pth" % (directory, filename, filename), weights_only=True))