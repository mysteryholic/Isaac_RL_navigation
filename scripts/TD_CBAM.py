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
    def __init__(self, input_shape, state_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()

        # # CNN Encoder for s_a (spatial lidar input)
        # self.cnn = nn.Sequential(
        #     nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),  # → (32,10,180)
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2), # → (64,5,90)
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), # → (64,3,45)
        #     nn.ReLU()
        # )

        # conv_out_size = 64 * 1* 15  # Flattened size after CNN

        self.l1=nn.Linear(state_dim+input_shape, 2*hidden_dim)
        self.l2=nn.Linear(2*hidden_dim, hidden_dim)
        self.l3=nn.Linear(hidden_dim, hidden_dim)
        self.l4=nn.Linear(hidden_dim, hidden_dim)
        self.l5=nn.Linear(hidden_dim, action_dim)

    def forward(self, s, s_a):  # s: state (B, state_dim), s_a: spatial input (B, 3, 10, 180)
        s=s.view(s.size(0), -1)
        s_a=s_a.view(s_a.size(0), -1)
        s=torch.cat((s,s_a), dim=1)

        h1 = F.relu(self.l1(s))
        h1=self.l2(h1)
        h1=F.relu(self.l3(h1))
        h1=F.relu(self.l4(h1))
        a =math.pi/4*torch.tanh(self.l5(h1))
        return a

class Critic(nn.Module):
    def __init__(self, input_shape, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()

        # # Shared CNN for both critics
        # def make_cnn():
        #     return nn.Sequential(
        #         nn.Conv2d(3, 32, 5, 1, 2),
        #         nn.ReLU(),
        #         nn.Conv2d(32, 64, 5, 2, 2),
        #         nn.ReLU(),
        #         nn.Conv2d(64, 64, 3, 2, 1),
        #         nn.ReLU()
        #     )

        # def init_weights(m):
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))

        # self.cnn1 = make_cnn()
        # self.cnn2 = make_cnn()
        # self.cnn1.apply(init_weights)  # cnn1에도 명시적 초기화 적용
        # self.cnn2.apply(init_weights)

        # conv_out_size = 64 * 1 * 15
        # Q1 network
        self.l1 = nn.Linear(state_dim + input_shape, hidden_dim)
        self.l2=nn.Linear(hidden_dim, hidden_dim)
        self.l3=nn.Linear(hidden_dim+action_dim, hidden_dim)
        self.l4=nn.Linear(hidden_dim, hidden_dim)
        self.l5=nn.Linear(hidden_dim, 1)

        # Q2 network
        self.l6 = nn.Linear(state_dim + input_shape, hidden_dim)
        self.l7=nn.Linear(hidden_dim, hidden_dim)
        self.l8=nn.Linear(hidden_dim+action_dim, hidden_dim)
        self.l9=nn.Linear(hidden_dim, hidden_dim)
        self.l10=nn.Linear(hidden_dim, 1)


    def forward(self, s, s_a, a):
        s=s.view(s.size(0), -1)
        s_a=s_a.view(s_a.size(0), -1)
        s=torch.cat((s,s_a), dim=1)

        h1 = F.relu(self.l1(s))
        h1=self.l2(h1)
        h1=torch.cat((h1,a), dim=1)
        h1=F.relu(self.l3(h1))
        h1=F.relu(self.l4(h1))
        q1 = self.l5(h1)
        h2 = F.relu(self.l6(s))
        h2=self.l7(h2)
        h2=torch.cat((h2,a), dim=1)
        h2=F.relu(self.l8(h2))
        h2=F.relu(self.l9(h2))
        q2 = self.l10(h2)

        return q1, q2
    
    def Q1(self, s, s_a, a):
        s=s.flatten(start_dim=1)
        s=torch.cat((s,s_a), dim=1)

        h1 = F.relu(self.l1(s))
        h1=self.l2(h1)
        h1=torch.cat((h1,a), dim=1)
        h1=F.relu(self.l3(h1))
        h1=F.relu(self.l4(h1))
        q1 = self.l5(h1)

        return q1


class TD3(object):
    def __init__(self, input_shape, num_inputs_add, action_space, args):
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.hidden_size = args.hidden_size
        self.policy_freq = args.policy_freq

        self.policy_type = args.policy
        self.count = 0
        state_shape=input_shape[2]*input_shape[3]*2
        state_shape=int(state_shape)



        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actor = Actor(state_shape, num_inputs_add, action_space, self.hidden_size).to(self.device)
        self.actor_target = Actor(state_shape, num_inputs_add, action_space, self.hidden_size).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.SGD(self.actor.parameters())

        self.critic = Critic(state_shape, num_inputs_add, action_space, self.hidden_size).to(self.device)
        self.critic_target = Critic(state_shape, num_inputs_add, action_space, self.hidden_size).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.SGD(self.critic.parameters())
        self.max_Q=float('-inf')
        

    def select_action(self, state, state_add):
        print("select_action using:")
        state = torch.FloatTensor(state).to(self.device)
        state = torch.clamp(state, 0.0, 5.1)
        state = state.unsqueeze(0)
        state_add = torch.FloatTensor(state_add).to(self.device)
        state_add = state_add.unsqueeze(0)
        return self.actor(state,state_add).cpu().data.numpy().flatten(),self.critic.Q1(state,state_add,self.actor(state,state_add)).cpu().data.numpy().flatten()
    
    
    def update_parameters(self, memory, args,expl_noise):
        max_Q=float('-inf')
        av_Q=0
        av_loss=0
        state_batch, state_batch_add, action_batch, reward_batch, next_state_batch, next_state_batch_add, mask_batch,weights = memory.sample(batch_size=args.batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        state_batch_add = torch.FloatTensor(state_batch_add).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        next_state_batch_add = torch.FloatTensor(next_state_batch_add).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device).view(-1, 1)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).view(-1, 1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).view(-1, 1)
        weights=torch.FloatTensor(weights).to(self.device).view(-1, 1)
        if torch.isnan(reward_batch).any():
            print("reward batch have a problem with nan")

        if torch.isnan(mask_batch).any():
            print("mask batch have a problem with nan")
        next_action = self.actor_target(next_state_batch, next_state_batch_add)
        
        noise = torch.randn_like(action_batch) * expl_noise
        noise=noise.clamp(-math.pi/8, math.pi/8)
        noise=noise.to(self.device)
        next_action = (next_action + noise).clamp(-math.pi/2, math.pi/2)

        target_Q1, target_Q2 = self.critic_target(next_state_batch, next_state_batch_add, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        av_Q += torch.mean(target_Q)
        max_Q = max(max_Q, torch.max(target_Q))
        if max_Q > self.max_Q:
            self.max_Q=max_Q

        target_Q = reward_batch + ((1 - mask_batch)* self.gamma * target_Q).detach()
        
        
        current_Q1, current_Q2 = self.critic(state_batch, state_batch_add, action_batch)
        if torch.isnan(target_Q).any():
            print("target Q have a problem with nan")
        if torch.isnan(current_Q1).any():
            print("Current Q1 have a problem with nan")
        if torch.isnan(current_Q2).any():
            print("Current Q2 have a problem with nan")
        loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        td_error = loss.mean()
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(),max_norm=1.0)
        self.critic_optimizer.step()
        if torch.isnan(loss).any():
            print("Loss have a problem with nan")
        if torch.isnan(td_error).any():
            print("TD Error have a problem with nan")

        if self.count % self.policy_freq == 0:
            actor_grad= self.critic.Q1(state_batch, state_batch_add, self.actor(state_batch, state_batch_add))
            actor_grad = -actor_grad.mean()
            self.actor_optimizer.zero_grad()
            actor_grad.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(),max_norm=1.0)
            self.actor_optimizer.step()
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        av_loss += loss
        av_loss /= (self.count + 1)
        self.count += 1
        memory.increase_beta()
        return av_loss, av_Q, max_Q,td_error
        

    # Save model parameters
    def save_checkpoint(self, directory, filename, i):
        torch.save(self.actor.state_dict(), "%s/%s/%s_actor_%d.pth" % (directory, filename, filename, i))
        torch.save(self.critic.state_dict(), "%s/%s/%s_critic_%d.pth" % (directory, filename, filename, i))
    
    # Load model parameters
    def load_checkpoint(self, directory, filename):
        self.actor.load_state_dict(torch.load("%s/%s/%s_actor.pth" % (directory, filename, filename), weights_only=True))
        self.critic.load_state_dict(torch.load("%s/%s/%s_critic.pth" % (directory, filename, filename), weights_only=True))