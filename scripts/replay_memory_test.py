import random
import numpy as np
from collections import deque
import os
import pickle
import torch
import copy


class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = deque()
        self.tmp=deque()
        self.goal=deque()
        self.position = 0
    
    def push(self, state, state_add, action, reward, next_state, next_state_add, done,goal_check):
        """
        상태 데이터를 버퍼에 저장
        - state: voxelized LiDAR 데이터 (torch.Tensor, 형태: [3, 10, 180])
        - state_add: 추가 상태 정보 (list 또는 numpy array)
        """
        

        # 상태가 튜플로 전달된 경우 확인 (디버깅 정보)
        if isinstance(state, tuple):
            print("Warning: state is already a tuple. Storing as-is.")
        
        # 그대로 저장 (object type)
        experience = (state, state_add, action, reward, next_state, next_state_add, done)
        self.buffer.append(experience)
        self.tmp.append(experience)

        # 버퍼 크기 관리
        if len(self.buffer) > self.capacity:
            self.buffer.popleft()
        if len(self.goal) > self.capacity:
            self.goal.popleft()

        if goal_check==True:
            self.goal.append(copy.deepcopy(self.tmp))
        else:
            self.tmp.clear()

    def sample(self, batch_size):
        """
        버퍼에서 배치 데이터 샘플링
        """
        batch = random.sample(self.buffer, batch_size)
        state_list, state_add_list, action_list, reward_list, next_state_list, next_state_add_list, done_list = zip(*batch)
        # 1. goal에서 모든 transition을 꺼냄 (flatten)
        all_goal_transitions = [transition for episode in self.goal for transition in episode]

        # 2. 최소한 batch_size만큼 있는 경우에만 샘플링
        goal_batch_size = min(len(all_goal_transitions), batch_size)
        if goal_batch_size > 0:
            goal_batch = random.sample(all_goal_transitions, goal_batch_size)
            g_state_list, g_state_add_list, g_action_list, g_reward_list, g_next_state_list, g_next_state_add_list, g_done_list = zip(*goal_batch)

            state_list += g_state_list
            state_add_list += g_state_add_list
            action_list += g_action_list
            reward_list += g_reward_list
            next_state_list += g_next_state_list
            next_state_add_list += g_next_state_add_list
            done_list += g_done_list

        try:
            # 상태(state)가 튜플인 경우를 처리
            processed_states = []
            for s in state_list:
                if isinstance(s, tuple):
                    # 튜플의 첫 번째 요소만 state로 사용 (두 번째 요소는 state_add)
                    if isinstance(s[0], torch.Tensor):
                        processed_states.append(s[0].clone().detach().float())
                    else:
                        processed_states.append(torch.tensor(s[0], dtype=torch.float32))
                elif isinstance(s, torch.Tensor):
                    processed_states.append(s.clone().detach().float())
                else:
                    processed_states.append(torch.tensor(s, dtype=torch.float32))
            
            # 다음 상태(next_state)도 동일한 방식으로 처리
            processed_next_states = []
            for s in next_state_list:
                if isinstance(s, tuple):
                    if isinstance(s[0], torch.Tensor):
                        processed_next_states.append(s[0].clone().detach().float())
                    else:
                        processed_next_states.append(torch.tensor(s[0], dtype=torch.float32))
                elif isinstance(s, torch.Tensor):
                    processed_next_states.append(s.clone().detach().float())
                else:
                    processed_next_states.append(torch.tensor(s, dtype=torch.float32))
            
            # 스택 처리
            state = torch.stack(processed_states)  # shape: [B, 3, 10, 180]
            next_state = torch.stack(processed_next_states)  # shape: [B, 3, 10, 180]
            
            # 추가 상태(state_add) 처리
            processed_state_adds = []
            for sa in state_add_list:
                if isinstance(sa, tuple) and len(sa) > 1:
                    # state가 튜플인 경우, 두 번째 요소가 실제 state_add일 수 있음
                    processed_state_adds.append(sa[1] if isinstance(sa, tuple) else sa)
                else:
                    processed_state_adds.append(sa)
            
            # 다음 추가 상태(next_state_add) 처리
            processed_next_state_adds = []
            for nsa in next_state_add_list:
                if isinstance(nsa, tuple) and len(nsa) > 1:
                    processed_next_state_adds.append(nsa[1] if isinstance(nsa, tuple) else nsa)
                else:
                    processed_next_state_adds.append(nsa)
            print(f"the shame of processed_next_state_adds: {len(processed_next_state_adds)}")
            # 나머지 데이터 처리
            state_add = torch.tensor(np.array(processed_state_adds), dtype=torch.float32)  # shape: [B, D]
            action = torch.tensor(np.array(action_list), dtype=torch.float32)  # shape: [B, action_dim]
            reward = torch.tensor(np.array(reward_list), dtype=torch.float32).unsqueeze(1)  # shape: [B, 1]
            next_state_add = torch.tensor(np.array(processed_next_state_adds), dtype=torch.float32)  # shape: [B, D]
            done = torch.tensor(np.array(done_list), dtype=torch.float32).unsqueeze(1)  # shape: [B, 1]

        except Exception as e:
            print("ReplayMemory sampling error:", str(e))
            # 자세한 디버깅 정보 출력
            print(f"Buffer size: {len(self.buffer)}")
            print(f"Batch size: {batch_size}")
            
            if len(state_list) > 0:
                print(f"First state type: {type(state_list[0])}")
                if isinstance(state_list[0], tuple):
                    print(f"First state is tuple with length: {len(state_list[0])}")
                    for i, item in enumerate(state_list[0]):
                        print(f"Tuple item {i} type: {type(item)}")
                        if isinstance(item, torch.Tensor):
                            print(f"Tensor shape: {item.shape}")
                else:
                    print(f"First state: {state_list[0]}")
            raise e
        
        return state, state_add, action, reward, next_state, next_state_add, done

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
