import random
import numpy as np
from collections import deque
import os
import pickle
import torch


class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = deque()
        self.priority = deque()
        self.alph=0.6
        self.beta=0.4
        self.epsilon=1e-6
        self.count=0
    
    def push(self, state, state_add, action, reward, next_state, next_state_add, done, td_error):
        """
        상태 데이터를 버퍼에 저장
        - td_error가 여러 값일 경우 평균을 사용하여 float으로 저장
        """
        # td_error를 float으로 변환
        if isinstance(reward, torch.Tensor):
            if reward.numel() == 1:
                reward = reward.item()
            else:
                reward = reward.mean().item()
        elif isinstance(reward, (list, np.ndarray)):
            reward = float(np.mean(td_error))
        
        # 경험 저장
        experience = (state, state_add, action, reward, next_state, next_state_add, done)
        self.buffer.append(experience)
        priority=-reward
        self.priority.append(priority)
        
        # 버퍼 크기 유지
        if len(self.buffer) > self.capacity:
            self.buffer.popleft()
            self.priority.popleft()
    def sample(self, batch_size):
        """
        버퍼에서 배치 데이터 샘플링
        """
        # priority를 numpy 배열로 안전하게 변환
        priority_array = np.array([float(p) for p in self.priority])
        
        # 안정성을 위해 최소값 보정
        priority_array += abs(np.min(priority_array))
        priorities = priority_array / np.sum(priority_array)
        priorities = np.clip(priorities, 0, 1)

        min_prob = np.min(priorities)
        max_weight = ((len(self.buffer) * min_prob)+self.epsilon) ** (-self.beta)
        weights = (((len(self.buffer) * priorities)+self.epsilon) ** (-self.beta)) / (max_weight+self.epsilon)
        weights = np.clip(weights, self.epsilon, 1)

        indices = np.random.choice(len(self.buffer), size=batch_size, p=priorities)
        batch = [self.buffer[i] for i in indices]
        state_list, state_add_list, action_list, reward_list, next_state_list, next_state_add_list, done_list = zip(*batch)
        weights = weights[indices]

        def process_state_list(state_list):
            processed = []
            for s in state_list:
                if isinstance(s, tuple):
                    s = s[0]  # tuple에서 첫 번째 요소만 사용
                processed.append(torch.tensor(s, dtype=torch.float32) if not isinstance(s, torch.Tensor) else s.clone().detach().float())
            return torch.stack(processed)

        state = process_state_list(state_list)
        next_state = process_state_list(next_state_list)
        state_add = torch.tensor(np.array(state_add_list), dtype=torch.float32)
        next_state_add = torch.tensor(np.array(next_state_add_list), dtype=torch.float32)
        action = torch.tensor(np.array(action_list), dtype=torch.float32)
        reward = torch.tensor(np.array(reward_list), dtype=torch.float32).unsqueeze(1)
        done = torch.tensor(np.array(done_list), dtype=torch.float32).unsqueeze(1)
        weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)

        return state, state_add, action, reward, next_state, next_state_add, done, weights

    def increase_beta(self):
        increase = (6 / 3000000) * self.count
        self.beta=min(self.beta+increase,1)

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()

    def save_buffer(self, env_name, suffix="", save_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        if save_path is None:
            save_path = "checkpoints/sac_buffer_{}_{}".format(env_name, suffix)
        print('Saving buffer to {}'.format(save_path))

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))

        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity
