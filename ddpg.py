import numpy as np
from datetime import datetime
from typing import List, Tuple, Dict
from copy import deepcopy
import torch
import os
from torch.nn import LazyLinear
import torch.nn.functional as F
import torch.optim as optim
import pickle

from utils import trunc_normal
import time

from IPython.display import clear_output
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pvt_graph import PVTGraph

class ReplayBuffer:
    

    def __init__(self, CktGraph, PVT_Graph, size: int, batch_size: int = 32):
        self.num_node_features = CktGraph.num_node_features
        self.action_dim = CktGraph.action_dim
        self.num_nodes = CktGraph.num_nodes
        self.pvt_dim = PVT_Graph.corner_dim
        self.num_corners = PVT_Graph.num_corners
        self.pvt_graph = PVT_Graph

        self.corner_buffers = {}  
        
        self.total_rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros([size], dtype=np.float32)
        
        self.max_size = size
        self.batch_size = batch_size
        self.ptr = 0
        self.size = 0
        self._init_corner_buffer()

    def _init_corner_buffer(self):
        
        for corner_idx , corner_name in enumerate(self.pvt_graph.pvt_corners.keys()):
            self.corner_buffers[corner_idx] = {
                'name': corner_name,
                'obs': np.zeros([self.max_size, self.num_nodes, self.num_node_features], dtype=np.float32),
                'next_obs': np.zeros([self.max_size, self.num_nodes, self.num_node_features], dtype=np.float32),
                'info': np.zeros([self.max_size], dtype=object),
                'reward': np.zeros([self.max_size], dtype=np.float32),
                
                'pvt_state': np.zeros([self.max_size, self.num_corners, self.pvt_dim], dtype=np.float32),
                'next_pvt_state': np.zeros([self.max_size, self.num_corners, self.pvt_dim], dtype=np.float32),
                
                'action': np.zeros([self.max_size, self.action_dim], dtype=np.float32),
                'corner_indices': [],  
                'attention_weights': np.zeros([self.max_size, self.num_corners], dtype=np.float32),
                
                'total_reward': np.zeros([self.max_size], dtype=np.float32),
                'done': np.zeros([self.max_size], dtype=np.float32),
                
                'ptr': 0,  
                'size': 0  
            }

    def store(
        self,
        pvt_state: np.ndarray,
        action: np.ndarray,
        results_dict: dict,
        next_pvt_state: np.ndarray,
        corner_indices: list,
        attention_weights: np.ndarray,
        total_reward: float,
        done: bool,
    ):
        
        if len(attention_weights) != self.num_corners:
            attention_weights = np.pad(attention_weights, (0, self.num_corners - len(attention_weights)))
        
        for corner_idx, result in results_dict.items():
            buffer = self.corner_buffers[corner_idx]
            ptr = buffer['ptr']
            
            buffer['obs'][ptr] = result['observation']
            buffer['next_obs'][ptr] = result['observation']  
            buffer['info'][ptr] = result['info']
            buffer['reward'][ptr] = result['reward']
            
            buffer['pvt_state'][ptr] = pvt_state
            buffer['next_pvt_state'][ptr] = next_pvt_state
            buffer['action'][ptr] = action
            buffer['corner_indices'].append(corner_indices)
            buffer['attention_weights'][ptr] = attention_weights
            buffer['total_reward'][ptr] = total_reward
            buffer['done'][ptr] = done
            
            buffer['ptr'] = (ptr + 1) % self.max_size
            buffer['size'] = min(buffer['size'] + 1, self.max_size)

    def sample_corner_batch(self, corner_idx: int) -> Dict[str, np.ndarray]:
        
        if corner_idx not in self.corner_buffers:
            return None
            
        buffer = self.corner_buffers[corner_idx]
        size = buffer['size']
        if size < self.batch_size:
            return None
            
        idxs = np.random.choice(size, size=self.batch_size, replace=False)
        
        return {
            'obs': buffer['obs'][idxs],
            'next_obs': buffer['next_obs'][idxs],
            'info': buffer['info'][idxs],
            'reward': buffer['reward'][idxs],
            'pvt_state': buffer['pvt_state'][idxs],
            'next_pvt_state': buffer['next_pvt_state'][idxs],
            'action': buffer['action'][idxs],
            'corner_indices': [buffer['corner_indices'][i] for i in idxs],
            'attention_weights': buffer['attention_weights'][idxs],
            'total_reward': buffer['total_reward'][idxs],
            'done': buffer['done'][idxs]
        }


class DDPGAgent:
    
    def __init__(
        self,
        env,
        CktGraph,
        PVT_Graph,
        Actor,
        Critic,
        memory_size: int,
        batch_size: int,
        noise_sigma: float,
        noise_sigma_min: float,
        noise_sigma_decay: float,
        noise_type: str,
        gamma: float = 0.99,
        tau: float = 5e-3,
        initial_random_steps: int = 1e4,
        sample_num: int = 3,
        agent_folder: str = None,
        old = False
    ):
        super().__init__()
        
        self.noise_sigma = noise_sigma
        self.noise_sigma_min = noise_sigma_min
        self.noise_sigma_decay = noise_sigma_decay
        self.action_dim = CktGraph.action_dim
        self.env = env
        self.memory = ReplayBuffer(CktGraph, PVT_Graph, memory_size, batch_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.initial_random_steps = initial_random_steps
        self.sample_num = sample_num
        self.old = old

        self.episode = 0
        self.device = CktGraph.device
        print(self.device)
        self.actor = Actor.to(self.device)
        self.actor_target = deepcopy(self.actor)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic.to(self.device)
        self.critic_target = deepcopy(self.critic)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=3e-4, weight_decay=1e-4)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=3e-4, weight_decay=1e-4)
        self.transition = list()
        self.total_step = 0
        self.agent_folder = agent_folder

        self.noise_type = noise_type
        self.is_test = False

        self.plots_dir = 'plots'  
        self.plots_rewards_dir = 'plots_rewards'  

        if not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir )

        if not os.path.exists(self.plots_rewards_dir):
            os.makedirs(self.plots_rewards_dir)


        self.pvt_corners = {}  
        self.best_corner = None
        self.best_corner_reward = -float('inf')

        self.perf_norm_params = {
            'phase_margin': {'target': 60, 'scale': 60},
            'dcgain': {'target': 130, 'scale': 130},
            'PSRP': {'target': -80, 'scale': 80},
            'PSRN': {'target': -80, 'scale': 80},
            'cmrrdc': {'target': -80, 'scale': 80}, 
            'vos': {'target': 0.06e-3, 'scale': 0.06e-3},
            'TC': {'target': 10e-6, 'scale': 10e-6},
            'settlingTime': {'target': 1e-6, 'scale': 1e-6},
            'FOML': {'target': 160, 'scale': 160},
            'FOMS': {'target': 300, 'scale': 300},
            'Active_Area': {'target': 150, 'scale': 150},
            'Power': {'target': 0.3, 'scale': 0.3},
            'GBW': {'target': 1.2e6, 'scale': 1.2e6},
            'sr': {'target': 0.6, 'scale': 0.6}
        }

        
        self.pvt_norm_params = {
            'vdd': {'min': 1.62, 'max': 1.98},  
            'temp': {'min': -40, 'max': 125},   
            'process': {'min': 0, 'max': 1}     
        }

        self.corner_rewards_history = {}
        self.corner_colors = plt.cm.rainbow(np.linspace(0, 1, self.actor.pvt_graph.num_corners))  

        self.skipped_steps = 0

    def _normalize_pvt_graph_state(self, state: torch.Tensor) -> torch.Tensor:
        
        state = state.clone()  
        
        state[:, 6] = (state[:, 6] - self.pvt_norm_params['vdd']['min']) / (self.pvt_norm_params['vdd']['max'] - self.pvt_norm_params['vdd']['min'])
        state[:, 5] = (state[:, 5] - self.pvt_norm_params['temp']['min']) / (self.pvt_norm_params['temp']['max'] - self.pvt_norm_params['temp']['min'])
        
        
        state[:, 7] = torch.sigmoid((state[:, 7] - 45) / 15)  
        
        state[:, 8] = torch.sigmoid((state[:, 8] - 120) / 10)  
        state[:, 9] = torch.sigmoid((state[:, 9] + 80) / 10)  
        state[:, 10] = torch.sigmoid((state[:, 10] + 80) / 10)  
        state[:, 11] = torch.sigmoid((state[:, 11] + 80) / 10)  
        
        state[:, 12] = torch.sigmoid((-torch.log10(state[:, 12]) - 5) / 0.5)  
        state[:, 13] = torch.sigmoid((-torch.log10(state[:, 13]) - 6) / 0.5)  
        
        state[:, 14] = torch.sigmoid((-torch.log10(state[:, 14]) - 3) / 0.5)  
        
        state[:, 15] = torch.sigmoid((state[:, 15] - 150) / 10)   
        state[:, 16] = torch.sigmoid((state[:, 16] - 280) / 20) 
        
        state[:, 17] = torch.sigmoid((150 - state[:, 17]) / 15)  
        state[:, 18] = torch.sigmoid((0.3 - state[:, 18]) / 0.03)  

        state[:, 19] = torch.sigmoid((torch.log10(state[:, 19]) - 6) / 0.5)  
        state[:, 20] = torch.sigmoid((state[:, 20] - 0.5) / 0.1)  
        
        
        state[:, 21] = (state[:, 21] - (-10)) / (2 - (-10))  
        
        return state

    def select_action(self, pvt_graph_state: torch.Tensor) -> np.ndarray:
        
        normalized_state = self._normalize_pvt_graph_state(pvt_graph_state)
        
        if self.is_test == False:
            if self.total_step < self.initial_random_steps:
                print('*** Random actions ***')
                selected_action = np.random.uniform(-1, 1, self.action_dim)
            else:
                print(f'*** Actions with Noise sigma = {self.noise_sigma} ***')
                
                selected_action = self.actor(
                    normalized_state  
                ).detach().cpu().numpy()
                selected_action = selected_action.flatten()
                if self.noise_type == 'uniform':
                    selected_action = np.random.uniform(np.clip(
                        selected_action-self.noise_sigma, -1, 1), np.clip(selected_action+self.noise_sigma, -1, 1))

                if self.noise_type == 'truncnorm':
                    selected_action = trunc_normal(selected_action, self.noise_sigma)
                    selected_action = np.clip(selected_action, -1, 1)
                
                self.noise_sigma = max(
                    self.noise_sigma_min, self.noise_sigma*self.noise_sigma_decay)

        else:   
            selected_action = self.actor(
                normalized_state  
            ).detach().cpu().numpy()
            selected_action = selected_action.flatten()

        self.transition = [normalized_state, selected_action]
        return selected_action


            



    def update_model(self) -> torch.Tensor:
        
        start_time = time.time()
        print("*** Update the model by gradient descent. ***")
        

        corner_indices = self.corner_indices
        corner_batches = {}
        for corner_idx in corner_indices:
            batch = self.memory.sample_corner_batch(corner_idx)
            if batch is not None:
                corner_batches[corner_idx] = batch
        
        if not corner_batches:  
            print("*** no batch abort ***")
            return None, None
            
        critic_losses = []
        for corner_idx, batch in corner_batches.items():
            obs = torch.FloatTensor(batch['obs']).to(self.device)
            next_obs = torch.FloatTensor(batch['next_obs']).to(self.device)
            action = torch.FloatTensor(batch['action']).to(self.device)
            reward = torch.FloatTensor(batch['reward']).reshape(-1, 1).to(self.device)
            done = torch.FloatTensor(batch['done']).reshape(-1, 1).to(self.device)
            pvt_state = torch.FloatTensor(batch['pvt_state']).to(self.device)
            next_pvt_state = torch.FloatTensor(batch['next_pvt_state']).to(self.device)
            
            
            values = self.critic(obs, action)
            critic_loss = F.mse_loss(values, reward)
            critic_losses.append(critic_loss)
        
        total_critic_loss = sum(critic_losses) / len(critic_losses)
        self.critic_optimizer.zero_grad()
        total_critic_loss.backward()
        self.critic_optimizer.step()
        
        actor_losses = []
        for corner_idx, batch in corner_batches.items():
            obs = torch.FloatTensor(batch['obs']).to(self.device)
            pvt_state = torch.FloatTensor(batch['pvt_state']).to(self.device)
            
            attention_weights = []
            for i in range(len(batch['corner_indices'])):  
                corner_indices = np.array(batch['corner_indices'][i])  
                idx_positions = np.where(corner_indices == corner_idx)[0] 
                if len(idx_positions) > 0:
                    idx = idx_positions[0]
                    weight = batch['attention_weights'][i][idx]
                else:
                    print(f"Corner index {corner_idx} not found in batch {i}")
                attention_weights.append(weight)
            
            attention_weight = torch.FloatTensor(attention_weights).reshape(-1, 1).to(self.device)
            
            value = self.critic(obs, self.actor(pvt_state))
            actor_loss = -(attention_weight * value).mean()
            actor_losses.append(actor_loss)
        
        total_actor_loss = sum(actor_losses) / len(actor_losses)
        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        self.actor_optimizer.step()

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Total time taken for update: {elapsed_time:.4f} seconds")
                
        return total_actor_loss.data, total_critic_loss.data

    def train(self, num_steps: int, plotting_interval: int = 50,check_point_interval: int=500, continue_training: bool = False):
        
        self.is_test = False      

        if not continue_training:     

            for corner_name in self.actor.pvt_graph.pvt_corners.keys():
                self.corner_rewards_history[corner_name] = []

            results_dict= self.env.reset()
        
            for corner_idx, result in results_dict.items():
                self.actor.update_pvt_performance_r(
                    corner_idx,
                    result['info'],
                    result['reward']
                )
                
            pvt_graph_state = self.actor.pvt_graph.get_graph_features()
        
            self.actor_losses = []              
            self.critic_losses = []
            self.scores = []
            self.score = 0
            

            for idx, corner_name in enumerate(self.actor.pvt_graph.pvt_corners.keys()):
                if idx in results_dict:  
                    reward = results_dict[idx]['reward']
                else:  
                    reward = self.corner_rewards_history[corner_name][-1]
                self.corner_rewards_history[corner_name].append(reward)

            print(f'*** The progress of the PVT graph ***')
            print("\nPVT Graph Rewards:")
            for corner_idx, corner_name in enumerate(self.actor.pvt_graph.pvt_corners.keys()):
                reward = pvt_graph_state[corner_idx][21]  
                print(f"{corner_name}: reward = {reward:.4f}")
            print()

        else:
            saved_agent = self.load_agent(self.agent_folder)
            self.actor.pvt_graph = saved_agent.actor.pvt_graph
            self.env.reset()
            pvt_graph_state = self.actor.pvt_graph.get_graph_features()
            print("PVT图状态已恢复")
            self.total_step = saved_agent.total_step
            self.episode = saved_agent.episode
            self.noise_sigma = saved_agent.noise_sigma
            self.initial_random_steps = 0
            print(f"Traning from steps: {self.total_step}")


            if not self.old:
                self.actor_losses = saved_agent.actor_losses
                self.critic_losses = saved_agent.critic_losses
                self.scores = saved_agent.scores
                self.score = saved_agent.score
                self.corner_rewards_history = saved_agent.corner_rewards_history
                print(f'*** The progress of the PVT graph ***')
                print("\nHistory PVT Graph Rewards:")
                for corner_idx, corner_name in enumerate(self.actor.pvt_graph.pvt_corners.keys()):
                    reward = pvt_graph_state[corner_idx][21]  
                    print(f"{corner_name}: reward = {reward:.4f}")
                print()



            else:
                self.actor_losses = []
                self.critic_losses = []
                self.scores = []
                self.score = 0
                for corner_name in self.actor.pvt_graph.pvt_corners.keys():
                    self.corner_rewards_history[corner_name] = [-5]

        for step in range(1, num_steps + 1):
            self.total_step += 1
            print(f'*** Step: {self.total_step} | Episode: {self.episode} ***')

            action = self.select_action(pvt_graph_state)
            
            if self.total_step >= self.initial_random_steps:
                attention_weights, corner_indices = self.actor.sample_corners(num_samples=self.sample_num)
                self.corner_indices = corner_indices
                total = sum(attention_weights)
                
                normalized = [x/total for x in attention_weights]
                
                attention_weights =  normalized

                print(sum(attention_weights))  

                print(f'*** corner_indices: {corner_indices} ***')
                print(f'*** corner_weights: {attention_weights} ***')
                
            else:
                corner_indices = np.arange(self.actor.num_PVT)
                num_corners = len(corner_indices)
                attention_weights = torch.ones(num_corners, device=self.device) / num_corners  

            results_dict, reward_no, terminated, truncated, info = self.env.step((action, corner_indices))
            
            if results_dict is None:
                print("Warning: results_dict is None, skipping this step")
                self.skipped_steps += 1  
                continue
                
            for corner_idx, result in results_dict.items():
                self.actor.update_pvt_performance_r(
                    corner_idx,
                    result['info'],
                    result['reward']
                )
                
            for idx, corner_name in enumerate(self.actor.pvt_graph.pvt_corners.keys()):
                if idx in results_dict:  
                    reward = results_dict[idx]['reward']
                else:  
                    reward = self.corner_rewards_history[corner_name][-1]
                self.corner_rewards_history[corner_name].append(reward)

            pvt_graph_state = self.actor.pvt_graph.get_graph_features()
            normalized_state = self._normalize_pvt_graph_state(pvt_graph_state)

            total_reward = 0
            for weight, corner_idx in zip(attention_weights, corner_indices):
                reward = results_dict[corner_idx]['reward']
                total_reward += weight * reward

            print(f'*** total_reward: {total_reward} ***')
            print()
            print(f'*** The progress of the PVT graph ***')
            print("\nPVT Graph Rewards:")
            for corner_idx, corner_name in enumerate(self.actor.pvt_graph.pvt_corners.keys()):
                reward = pvt_graph_state[corner_idx][21]  
                print(f"{corner_name}: reward = {reward:.4f}")
            print()

            print("\nCorner indices sorted by reward:", end=' ')
            rewards = pvt_graph_state[:, 21]  
            sorted_indices = np.argsort(rewards)  
            for i, idx in enumerate(sorted_indices): print(f"{idx}", end=' ')
            print()


            if total_reward > 0:  
                terminated = True
            else:
                terminated = False

            self.score += total_reward

            if not self.is_test:
                self.transition += [
                    results_dict,
                    normalized_state,
                    corner_indices,
                    attention_weights,
                    total_reward,
                    terminated
                ]
                self.memory.store(*self.transition)
                
            if terminated or truncated:
                results_dict = self.env.reset()
                print(f"*** reset ***")
                for corner_idx, result in results_dict.items():
                    self.actor.update_pvt_performance_r(
                        corner_idx,
                        result['info'],
                        result['reward']
                    )
                
                pvt_graph_state = self.actor.pvt_graph.get_graph_features()
                self.episode += 1
                self.scores.append(self.score)
                self.score = 0
                print(f'*** The progress of the PVT graph ***')
                print("\nPVT Graph Rewards:")
                for corner_idx, corner_name in enumerate(self.actor.pvt_graph.pvt_corners.keys()):
                    reward = pvt_graph_state[corner_idx][21]  
                    print(f"{corner_name}: reward = {reward:.4f}")
                print()



            if  self.total_step > self.initial_random_steps:
                self.actor_loss, self.critic_loss = self.update_model()
                self.actor_losses.append(self.actor_loss)
                self.critic_losses.append(self.critic_loss)
            
            if self.total_step % plotting_interval == 0:
                self._plot(
                    self.total_step,
                    self.scores,
                    self.actor_losses,
                    self.critic_losses,
                )
                self.plot_corner_rewards()

            if self.total_step % check_point_interval == 0:
                self.save_check_point()
            

        print(f"\nTraining completed:")
        print(f"Total steps: {self.total_step}")
        print(f"Skipped steps: {self.skipped_steps}")
        
        self.env.close()

    def test(self):
        
        self.is_test = True
        results_dict = self.env.reset()
        
        for corner_idx, result in results_dict.items():
            self.actor.update_pvt_performance_r(
                corner_idx,
                result['info'],
                result['reward']
            )
            
        pvt_graph_state = self.actor.pvt_graph.get_graph_features()
        truncated = False
        terminated = False
        score = 0
        
        performance_list = []
        while not (truncated or terminated):    
            action = self.select_action(pvt_graph_state)
            
            attention_weights, corner_indices = self.actor.sample_corners(num_samples=self.sample_num)
            
            results_dict, terminated, truncated = self.env.step((action, corner_indices))
            
            for corner_idx, result in results_dict.items():
                self.actor.update_pvt_performance_r(
                    corner_idx,
                    result['info'],
                    result['reward']
                )
            
            performance_list.append({
                'action': action,
                'corner_indices': corner_indices,
                'attention_weights': attention_weights,
                'results': {idx: {
                    'info': result['info'],
                    'reward': result['reward']
                } for idx, result in results_dict.items()}
            })
            
            total_reward = 0
            for weight, corner_idx in zip(attention_weights, corner_indices):
                reward = results_dict[corner_idx]['reward']
                total_reward += weight * reward
            
            pvt_graph_state = self.actor.pvt_graph.get_graph_features()
            score += total_reward
            
        print(f"score: {score}")
        print("Performance in each corner:")
        for corner_idx, result in results_dict.items():
            print(f"Corner {corner_idx}:")
            print(f"Info: {result['info']}")
            print(f"Reward: {result['reward']}")
            
        self.env.close()
        return performance_list

    def _target_soft_update(self):
        
        tau = self.tau      
        for t_param, l_param in zip(
            self.actor_target.parameters(), self.actor.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

        for t_param, l_param in zip(
            self.critic_target.parameters(), self.critic.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

    def _plot(
        self,
        step: int,
        scores: List[float],
        actor_losses: List[float],
        critic_losses: List[float],
    ):
        

        
        def subplot(loc: int, title: str, values: List[float]):
            plt.subplot(loc)
            plt.title(title)
            plt.plot(values)

        subplot_params = [
            (131, f"step {step}", scores),
            (132, "actor_loss", actor_losses),
            (133, "critic_loss", critic_losses),
        ]

        plt.figure(figsize=(30, 5))
        for loc, title, values in subplot_params:
            subplot(loc, title, values)
        
        filename = os.path.join(self.plots_dir, f"step_{step:06d}.png")
        plt.savefig(filename)
        plt.close('all')  

    def plot_corner_rewards(self):
        
        plt.figure(figsize=(10, 6))
        plt.title('PVT Corner Rewards vs Steps')
        plt.xlabel('Step')
        plt.ylabel('Reward')
        
        for idx, (corner_name, rewards) in enumerate(self.corner_rewards_history.items()):
            plt.plot(range(len(rewards)), rewards, 
                    label=corner_name, 
                    color=self.corner_colors[idx],
                    alpha=0.7)
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        
        filename = os.path.join(self.plots_rewards_dir, f"corner_rewards_step_{self.total_step:06d}.png")
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

    def load_replay_buffer(self, buffer_path):
        
        if not os.path.exists(buffer_path):
            print(f"No saved buffer found at {buffer_path}")
            return
            
        print(f"\nLoading replay buffer from {buffer_path}")
        
        print("\nCurrent buffer corners:")
        for idx, corner_name in enumerate(self.actor.pvt_graph.pvt_corners.keys()):
            print(f"Index {idx}: {corner_name}")
            
        with open(buffer_path, 'rb') as f:
            saved_buffer = pickle.load(f)
        
        print("\nSaved buffer corners:")
        for corner_idx, saved_corner_buffer_idx in enumerate(saved_buffer.corner_buffers.keys()):
            print(f"Index {corner_idx}: {saved_corner_buffer_idx} : {saved_buffer.corner_buffers[saved_corner_buffer_idx]['name']}")
            
        for corner_idx, saved_corner_buffer in saved_buffer.corner_buffers.items():
            buffer = self.memory.corner_buffers[corner_idx]
            
            buffer['obs'][:saved_corner_buffer['size']] = saved_corner_buffer['obs'][:saved_corner_buffer['size']]
            buffer['next_obs'][:saved_corner_buffer['size']] = saved_corner_buffer['next_obs'][:saved_corner_buffer['size']]
            buffer['info'][:saved_corner_buffer['size']] = saved_corner_buffer['info'][:saved_corner_buffer['size']]
            buffer['reward'][:saved_corner_buffer['size']] = saved_corner_buffer['reward'][:saved_corner_buffer['size']]
            buffer['pvt_state'][:saved_corner_buffer['size']] = saved_corner_buffer['pvt_state'][:saved_corner_buffer['size']]
            buffer['next_pvt_state'][:saved_corner_buffer['size']] = saved_corner_buffer['next_pvt_state'][:saved_corner_buffer['size']]
            buffer['action'][:saved_corner_buffer['size']] = saved_corner_buffer['action'][:saved_corner_buffer['size']]
            buffer['corner_indices'] = saved_corner_buffer['corner_indices'][:saved_corner_buffer['size']]
            buffer['attention_weights'][:saved_corner_buffer['size']] = saved_corner_buffer['attention_weights'][:saved_corner_buffer['size']]
            buffer['total_reward'][:saved_corner_buffer['size']] = saved_corner_buffer['total_reward'][:saved_corner_buffer['size']]
            buffer['done'][:saved_corner_buffer['size']] = saved_corner_buffer['done'][:saved_corner_buffer['size']]
            
            buffer['ptr'] = saved_corner_buffer['ptr']
            buffer['size'] = saved_corner_buffer['size']
            
            print(f"Loaded data for corner {corner_idx} ({saved_corner_buffer['name']}) to ({buffer['name']})")
            print(f"  Size: {buffer['size']}")
            
        print(f"\nSuccessfully loaded replay buffer with {len(saved_buffer.corner_buffers)} corners")

    def load_agent(self,agent_folder):
        actor_weights = [f for f in os.listdir(agent_folder) if f.startswith('Actor')]
        if actor_weights:
            actor_path = os.path.join(agent_folder, actor_weights[0])
            actor_state_dict = torch.load(actor_path)

        critic_weights = [f for f in os.listdir(agent_folder) if f.startswith('Critic')]
        if critic_weights:
            critic_path = os.path.join(agent_folder, critic_weights[0])
            critic_state_dict = torch.load(critic_path)

        memory_files = [f for f in os.listdir(agent_folder) if f.startswith('memory')]
        if memory_files:
            memory_path = os.path.join(agent_folder, memory_files[0])

        agent_files = [f for f in os.listdir(agent_folder) if f.startswith('DDPGAgent')]
        if agent_files:
            agent_path = os.path.join(agent_folder, agent_files[0])
            with open(agent_path, 'rb') as f:
                saved_agent = pickle.load(f)

        if 'actor_state_dict' in locals():
            self.actor.load_state_dict(actor_state_dict)
            print("Actor权重已加载")
            
        if 'critic_state_dict' in locals():
            self.critic.load_state_dict(critic_state_dict)
            print("Critic权重已加载")
            
        if 'memory_path' in locals():
            self.load_replay_buffer(memory_path)
            print("Memory已加载")
             
        print("Agent加载完成，继续训练...\n")
        return saved_agent
    

    def save_check_point(self):
        print("********Replay the best results********")
        memory = self.memory
        best_reward = float('-inf')
        best_action = None
        best_corner = None


        for corner_idx, buffer in memory.corner_buffers.items():
            rewards = buffer['total_reward'][:buffer['size']]
            if len(rewards) > 0:
                max_reward = np.max(rewards)
                if max_reward > best_reward:
                    best_reward = max_reward
                    idx = np.argmax(rewards)
                    best_action = buffer['action'][idx]
                    best_corner = corner_idx

        if best_action is not None:
            results_dict, flag, terminated, truncated, info = self.env.step(
                (best_action, np.arange(self.actor.num_PVT), True)
            )


        PWD = os.getcwd()
        num_steps = self.total_step
        num_corners = self.actor.num_PVT

        current_time = datetime.now().strftime('%m-%d_%H-%M')
        folder_name = f"check_point_{current_time}_steps{num_steps}_corners-{num_corners}_reward-{best_reward:.2f}"

        save_dir = os.path.join(PWD, 'saved_results', folder_name)
        
        os.makedirs(save_dir, exist_ok=True)

        results_file_name = f"opt_result_{current_time}_steps{num_steps}_corners-{num_corners}_reward-{best_reward:.2f}"
        results_path = os.path.join(save_dir, results_file_name)
        with open(results_path, 'w') as f:
            f.writelines(self.env.unwrapped.get_saved_results)  

        model_weight_actor = self.actor.state_dict()
        save_name_actor = f"Actor_weight_{current_time}_steps{num_steps}_corners-{num_corners}_reward-{best_reward:.2f}.pth"
        
        model_weight_critic = self.critic.state_dict()
        save_name_critic = f"Critic_weight_{current_time}_steps{num_steps}_corners-{num_corners}_reward-{best_reward:.2f}.pth"
        
        torch.save(model_weight_actor, os.path.join(save_dir, save_name_actor))
        torch.save(model_weight_critic, os.path.join(save_dir, save_name_critic))
        print("Actor and Critic weights have been saved!")

        memory_path = os.path.join(save_dir, f'memory_{current_time}_steps{num_steps}_corners-{num_corners}_reward-{best_reward:.2f}.pkl')
        with open(memory_path, 'wb') as memory_file:
            pickle.dump(memory, memory_file)

        agent_path = os.path.join(save_dir, f'DDPGAgent_{current_time}_steps{num_steps}_corners-{num_corners}_reward-{best_reward:.2f}.pkl')
        with open(agent_path, 'wb') as agent_file:
            pickle.dump(self, agent_file)
            
        print(f"checkpoint have been saved in: {save_dir}")