import gymnasium as gym
import tarware
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim

from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv

from torchdiffeq import odeint
import wandb
from datetime import datetime
import numpy as np
from collections import deque
import random
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

parser = ArgumentParser(description="Run tests with vector environments on WarehouseEnv", formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument(
        "--num_episodes",
        default=1000,
        type=int,
        help="The seed to run with"
    )
parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="The seed to run with"
    )

parser.add_argument(
        "--render",
        default=False,
        action='store_true',
    )

args = parser.parse_args()

def info_statistics(infos, global_episode_return, episode_returns):
    _total_deliveries = 0
    _total_clashes = 0
    _total_stuck = 0
    for info in infos:
        _total_deliveries += info["shelf_deliveries"]
        _total_clashes += info["clashes"]
        _total_stuck += info["stucks"]
        info["total_deliveries"] = _total_deliveries
        info["total_clashes"] = _total_clashes
        info["total_stuck"] = _total_stuck
    last_info = infos[-1]
    last_info["episode_length"] = len(infos)
    last_info["global_episode_return"] = global_episode_return
    last_info["episode_returns"] = episode_returns
    return last_info

class HeteroGraphODENetwork(nn.Module):
    """Heterogeneous Graph Neural ODE for MARL """
    
    def __init__(self, node_dims, action_size, hidden_dim=64, num_layers=2, ode_hidden_dim=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ode_hidden_dim = ode_hidden_dim
        
        # Node type dimensions
        self.agv_dim = node_dims['agv']
        self.picker_dim = node_dims['picker'] 
        self.location_dim = node_dims['location']
        
        # Initial node embeddings
        self.agv_embedding = nn.Linear(self.agv_dim, hidden_dim)
        self.picker_embedding = nn.Linear(self.picker_dim, hidden_dim)
        self.location_embedding = nn.Linear(self.location_dim, hidden_dim)
        
        # Heterogeneous GNN layers for initial processing
        self.hetero_convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {
                ('agv', 'targets', 'location'): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                ('location', 'is targeted by', 'agv'): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                ('agv', 'communicates', 'agv'): SAGEConv(hidden_dim, hidden_dim),
                ('picker', 'manages', 'location'): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                ('agv', 'cooperates with', 'picker'): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                ('picker', 'helps', 'agv'): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
            }
            self.hetero_convs.append(HeteroConv(conv_dict, aggr='mean'))
        
        # ODE Function Network for each agent type
        self.ode_func_agv = ODEFunction(hidden_dim, ode_hidden_dim)
        self.ode_func_picker = ODEFunction(hidden_dim, ode_hidden_dim)

        # Action heads for each agent type
        self.agv_action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_size)
        )
        
        self.picker_action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_size)
        )
        
    def forward(self, hetero_data, integration_time=1.0):
        # 1. Initial Embeddings
        x_dict = {
            'agv': self.agv_embedding(hetero_data['agv'].x),
            'picker': self.picker_embedding(hetero_data['picker'].x),
            'location': self.location_embedding(hetero_data['location'].x)
        }
        
        # 2. GNN Processing for Collaboration Context
        # GNN을 통해 주변 노드 정보를 취합하여 협업을 위한 '현재 상황'을 파악합니다.
        for conv in self.hetero_convs:
            x_dict = conv(x_dict, hetero_data.edge_index_dict)
            x_dict = {key: torch.relu(x) for key, x in x_dict.items()}
        
        # 3. Type-Specific Neural ODE Application
        # 각 에이전트 타입에 맞는 ODE 함수로 '미래 상태'를 예측합니다.
        # 이제 더 이상 모든 임베딩을 합치지 않습니다.
        t = torch.tensor([0., integration_time], dtype=torch.float32, device=x_dict['agv'].device)
        
        evolved_agv_embeddings = odeint(self.ode_func_agv, x_dict['agv'], t, method='euler')[-1] # 'dopri5'
        evolved_picker_embeddings = odeint(self.ode_func_picker, x_dict['picker'], t, method='euler')[-1] # 'dopri5'
        
        # Location 노드는 동적인 상태 변화가 없으므로 ODE를 적용하지 않고 GNN의 결과만 사용합니다.
        final_location_embeddings = x_dict['location']
        
        # 4. Generate Action Values from Evolved States
        # 예측된 미래 상태를 기반으로 최적의 행동(Q-value)을 결정합니다.
        agv_q_values = self.agv_action_head(evolved_agv_embeddings)
        picker_q_values = self.picker_action_head(evolved_picker_embeddings)
        
        return {
            'agv_q_values': agv_q_values,
            'picker_q_values': picker_q_values,
            'agv_embeddings': evolved_agv_embeddings,
            'picker_embeddings': evolved_picker_embeddings,
            'location_embeddings': final_location_embeddings
        }

class ODEFunction(nn.Module):
    """ODE Function for continuous dynamics"""
    
    def __init__(self, hidden_dim, ode_hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, ode_hidden_dim),
            nn.Tanh(),
            nn.Linear(ode_hidden_dim, ode_hidden_dim),
            nn.Tanh(),
            nn.Linear(ode_hidden_dim, hidden_dim)
        )
        
    def forward(self, t, x):
        return self.net(x)

class TypeSpecificActor(nn.Module):
    """Actor network for specific agent type (AGV or Picker)"""
    
    def __init__(self, embedding_dim, action_size, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_size)
        )
        
    def forward(self, embeddings):
        """
        Args:
            embeddings: [num_agents_of_type, embedding_dim]
        Returns:
            action_logits: [num_agents_of_type, action_size]
        """
        return self.network(embeddings)
    
    def get_action_probs(self, embeddings, action_masks=None):
        """Get action probabilities with masking"""
        logits = self.forward(embeddings)
        
        if action_masks is not None:
            # Apply action masking
            logits = logits.masked_fill(~action_masks.bool(), -1e9)
        
        return F.softmax(logits, dim=-1)
    
    def sample_actions(self, embeddings, action_masks=None):
        """Sample actions and return log probabilities"""
        probs = self.get_action_probs(embeddings, action_masks)
        dist = Categorical(probs)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        return actions, log_probs

class TypeSpecificCritic(nn.Module):
    """Critic network for specific agent type (takes global type state + all type actions)"""
    
    def __init__(self, num_agents_of_type, embedding_dim, action_size, hidden_dim=128):
        super().__init__()
        self.num_agents = num_agents_of_type
        self.action_size = action_size
        
        # Input: all embeddings + all actions (one-hot)
        input_dim = num_agents_of_type * (embedding_dim + action_size)
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_agents_of_type)  # Q-value for each agent of this type
        )
        
    def forward(self, embeddings, actions):
        """
        Args:
            embeddings: [num_agents_of_type, embedding_dim]
            actions: [num_agents_of_type] - discrete actions
        Returns:
            q_values: [num_agents_of_type] - Q-value for each agent
        """
        # Convert actions to one-hot
        actions_one_hot = F.one_hot(actions.long(), num_classes=self.action_size).float()
        
        # Concatenate embeddings and actions for all agents of this type
        # Shape: [embedding_dim + action_size] for each agent, then flatten
        agent_features = torch.cat([embeddings, actions_one_hot], dim=1)  # [num_agents, embedding_dim + action_size]
        global_input = agent_features.view(-1)  # Flatten to [num_agents * (embedding_dim + action_size)]
        
        # Add batch dimension if needed
        if global_input.dim() == 1:
            global_input = global_input.unsqueeze(0)
        
        q_values = self.network(global_input)  # [1, num_agents]
        return q_values.squeeze(0)  # [num_agents]

class HeteroGraphActorCriticNetwork(nn.Module):
    """Graph-based Actor-Critic Network using existing Graph ODE"""
    
    def __init__(self, graph_ode_network, num_agvs, num_pickers, action_size, hidden_dim=64):
        super().__init__()
        self.graph_network = graph_ode_network  # Your existing HeteroGraphODENetwork
        self.num_agvs = num_agvs
        self.num_pickers = num_pickers
        self.action_size = action_size
        self.hidden_dim = hidden_dim
        
        # Get embedding dimension from graph network
        embedding_dim = graph_ode_network.hidden_dim
        
        # Type-specific actors
        self.agv_actor = TypeSpecificActor(embedding_dim, action_size, hidden_dim)
        self.picker_actor = TypeSpecificActor(embedding_dim, action_size, hidden_dim)
        
        # Type-specific critics
        self.agv_critic = TypeSpecificCritic(num_agvs, embedding_dim, action_size, hidden_dim)
        self.picker_critic = TypeSpecificCritic(num_pickers, embedding_dim, action_size, hidden_dim)
        
    def forward(self, hetero_data, integration_time=1.0):
        """Get embeddings from graph network"""
        graph_output = self.graph_network(hetero_data, integration_time)
        return {
            'agv_embeddings': graph_output['agv_embeddings'],
            'picker_embeddings': graph_output['picker_embeddings'],
            'location_embeddings': graph_output['location_embeddings']
        }
    
    def get_actions_and_probs(self, hetero_data, valid_action_masks, training=True):
        """Get actions and probabilities for all agents"""
        graph_output = self.forward(hetero_data)
        
        agv_embeddings = graph_output['agv_embeddings']  # [num_agvs, embedding_dim]
        picker_embeddings = graph_output['picker_embeddings']  # [num_pickers, embedding_dim]
        
        # Split action masks
        agv_masks = valid_action_masks[:self.num_agvs] if valid_action_masks is not None else None
        picker_masks = valid_action_masks[self.num_agvs:] if valid_action_masks is not None else None
        
        if training:
            # Sample actions during training
            agv_actions, agv_log_probs = self.agv_actor.sample_actions(agv_embeddings, agv_masks)
            picker_actions, picker_log_probs = self.picker_actor.sample_actions(picker_embeddings, picker_masks)
            
            # Combine results
            all_actions = torch.cat([agv_actions, picker_actions])
            all_log_probs = torch.cat([agv_log_probs, picker_log_probs])
            
            return all_actions, all_log_probs, graph_output
        else:
            # Greedy actions during evaluation
            agv_probs = self.agv_actor.get_action_probs(agv_embeddings, agv_masks)
            picker_probs = self.picker_actor.get_action_probs(picker_embeddings, picker_masks)
            
            agv_actions = torch.argmax(agv_probs, dim=-1)
            picker_actions = torch.argmax(picker_probs, dim=-1)
            
            all_actions = torch.cat([agv_actions, picker_actions])
            return all_actions, None, graph_output
    
    def compute_counterfactual_advantage(self, graph_output, agent_type, agent_idx, actions, critic_type):
        """
        Compute COMA counterfactual advantage for specific agent
        
        Args:
            graph_output: Output from graph network
            agent_type: 'agv' or 'picker'
            agent_idx: Index within the agent type
            actions: Actions taken by all agents of this type
            critic_type: 'agv_critic' or 'picker_critic'
        """
        if agent_type == 'agv':
            embeddings = graph_output['agv_embeddings']
            critic = self.agv_critic
        else:
            embeddings = graph_output['picker_embeddings'] 
            critic = self.picker_critic
            
        # Current Q-value
        current_q = critic(embeddings, actions)[agent_idx]
        
        # Compute counterfactual baseline
        counterfactual_q_values = []
        
        for alt_action in range(self.action_size):
            # Create counterfactual action vector
            cf_actions = actions.clone()
            cf_actions[agent_idx] = alt_action
            
            # Get Q-value for counterfactual action
            cf_q = critic(embeddings, cf_actions)[agent_idx]
            counterfactual_q_values.append(cf_q)
        
        counterfactual_q_values = torch.stack(counterfactual_q_values)  # [action_size]
        
        # Get current policy for this agent
        if agent_type == 'agv':
            agent_embedding = embeddings[agent_idx:agent_idx+1]
            action_probs = self.agv_actor.get_action_probs(agent_embedding).squeeze(0)
        else:
            agent_embedding = embeddings[agent_idx:agent_idx+1]
            action_probs = self.picker_actor.get_action_probs(agent_embedding).squeeze(0)
        
        # Baseline = expected Q-value under current policy
        baseline = torch.sum(action_probs * counterfactual_q_values)
        
        # Advantage = Q(current action) - baseline
        advantage = current_q - baseline
        
        return advantage

class TypeLevelCOMA:
    """Type-Level COMA Agent"""
    
    def __init__(self, graph_ode_network, num_agvs, num_pickers, action_size, 
                 lr_actor=1e-3, lr_critic=1e-3, gamma=0.99, memory_size=10000):
        
        self.num_agvs = num_agvs
        self.num_pickers = num_pickers
        self.action_size = action_size
        self.gamma = gamma
        
        # Actor-Critic network
        self.ac_network = HeteroGraphActorCriticNetwork(
            graph_ode_network, num_agvs, num_pickers, action_size
        )
        
        # Optimizers for each component
        self.agv_actor_optimizer = optim.Adam(self.ac_network.agv_actor.parameters(), lr=lr_actor)
        self.picker_actor_optimizer = optim.Adam(self.ac_network.picker_actor.parameters(), lr=lr_actor)
        self.agv_critic_optimizer = optim.Adam(self.ac_network.agv_critic.parameters(), lr=lr_critic)
        self.picker_critic_optimizer = optim.Adam(self.ac_network.picker_critic.parameters(), lr=lr_critic)
        
        # Experience replay
        self.memory = deque(maxlen=memory_size)
        
    def act(self, hetero_data, valid_action_masks, training=True):
        """Choose actions for all agents"""
        with torch.no_grad():
            actions, log_probs, graph_output = self.ac_network.get_actions_and_probs(
                hetero_data, valid_action_masks, training
            )
        
        if training:
            return actions.cpu().numpy().tolist(), log_probs, graph_output
        else:
            return actions.cpu().numpy().tolist(), None, graph_output
    
    def remember(self, hetero_data, actions, rewards, next_hetero_data, dones):
        """Store experience in replay buffer"""
        self.memory.append((hetero_data, actions, rewards, next_hetero_data, dones))
    
    def replay(self, batch_size=32):
        """Train the networks using COMA"""
        if len(self.memory) < batch_size:
            return {}
        
        batch = random.sample(self.memory, batch_size)
        
        total_agv_actor_loss = 0
        total_picker_actor_loss = 0
        total_agv_critic_loss = 0
        total_picker_critic_loss = 0
        
        for experience in batch:
            hetero_data, actions, rewards, next_hetero_data, dones = experience
            
            # Convert to tensors
            actions_tensor = torch.tensor(actions, dtype=torch.long)
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
            dones_tensor = torch.tensor(dones, dtype=torch.float32)
            
            # Split actions and rewards by type
            agv_actions = actions_tensor[:self.num_agvs]
            picker_actions = actions_tensor[self.num_agvs:]
            agv_rewards = rewards_tensor[:self.num_agvs]
            picker_rewards = rewards_tensor[self.num_agvs:]
            agv_dones = dones_tensor[:self.num_agvs]
            picker_dones = dones_tensor[self.num_agvs:]
            
            # Get current and next embeddings
            current_output = self.ac_network.forward(hetero_data)
            
            with torch.no_grad():
                next_output = self.ac_network.forward(next_hetero_data)
            
            # ============ AGV UPDATES ============
            if self.num_agvs > 0:
                # AGV Critic Update
                current_agv_q = self.ac_network.agv_critic(current_output['agv_embeddings'], agv_actions)
                
                with torch.no_grad():
                    next_agv_q = self.ac_network.agv_critic(next_output['agv_embeddings'], agv_actions[:self.num_agvs])
                    agv_targets = agv_rewards + self.gamma * next_agv_q * (1 - agv_dones)
                
                agv_critic_loss = F.mse_loss(current_agv_q, agv_targets)
                
                self.agv_critic_optimizer.zero_grad()
                agv_critic_loss.backward(retain_graph=True)
                self.agv_critic_optimizer.step()
                
                total_agv_critic_loss += agv_critic_loss.item()
                
                # AGV Actor Update
                agv_actor_loss = 0
                for agv_idx in range(self.num_agvs):
                    advantage = self.ac_network.compute_counterfactual_advantage(
                        current_output, 'agv', agv_idx, agv_actions, 'agv_critic'
                    )
                    
                    # Get log probability of taken action
                    agv_embedding = current_output['agv_embeddings'][agv_idx:agv_idx+1]
                    action_probs = self.ac_network.agv_actor.get_action_probs(agv_embedding)
                    taken_action = agv_actions[agv_idx]
                    log_prob = torch.log(action_probs[0, taken_action] + 1e-8)
                    
                    agv_actor_loss += -log_prob * advantage.detach()
                
                agv_actor_loss = agv_actor_loss / self.num_agvs
                
                self.agv_actor_optimizer.zero_grad()
                agv_actor_loss.backward(retain_graph=True)
                self.agv_actor_optimizer.step()
                
                total_agv_actor_loss += agv_actor_loss.item()
            
            # ============ PICKER UPDATES ============
            if self.num_pickers > 0:
                # Picker Critic Update
                current_picker_q = self.ac_network.picker_critic(current_output['picker_embeddings'], picker_actions)
                
                with torch.no_grad():
                    next_picker_q = self.ac_network.picker_critic(next_output['picker_embeddings'], picker_actions)
                    picker_targets = picker_rewards + self.gamma * next_picker_q * (1 - picker_dones)
                
                picker_critic_loss = F.mse_loss(current_picker_q, picker_targets)
                
                self.picker_critic_optimizer.zero_grad()
                picker_critic_loss.backward(retain_graph=True)
                self.picker_critic_optimizer.step()
                
                total_picker_critic_loss += picker_critic_loss.item()
                
                # Picker Actor Update
                picker_actor_loss = 0
                for picker_idx in range(self.num_pickers):
                    advantage = self.ac_network.compute_counterfactual_advantage(
                        current_output, 'picker', picker_idx, picker_actions, 'picker_critic'
                    )
                    
                    # Get log probability of taken action
                    picker_embedding = current_output['picker_embeddings'][picker_idx:picker_idx+1]
                    action_probs = self.ac_network.picker_actor.get_action_probs(picker_embedding)
                    taken_action = picker_actions[picker_idx]
                    log_prob = torch.log(action_probs[0, taken_action] + 1e-8)
                    
                    picker_actor_loss += -log_prob * advantage.detach()
                
                picker_actor_loss = picker_actor_loss / self.num_pickers
                
                self.picker_actor_optimizer.zero_grad()
                picker_actor_loss.backward()
                self.picker_actor_optimizer.step()
                
                total_picker_actor_loss += picker_actor_loss.item()
        
        return {
            'agv_actor_loss': total_agv_actor_loss / batch_size,
            'picker_actor_loss': total_picker_actor_loss / batch_size,
            'agv_critic_loss': total_agv_critic_loss / batch_size,
            'picker_critic_loss': total_picker_critic_loss / batch_size
        }

class SimpleIndependentDQN:
    """Simple Independent Q-Learning using Graph ODE for feature extraction"""
    
    def __init__(self, graph_ode_network, num_agvs, num_pickers, action_size, 
                 lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, 
                 epsilon_min=0.01, memory_size=10000):
        
        self.graph_network = graph_ode_network
        self.num_agvs = num_agvs
        self.num_pickers = num_pickers
        self.num_agents = num_agvs + num_pickers
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Target network (same as main network)
        self.target_network = HeteroGraphODENetwork(
            {'agv': graph_ode_network.agv_dim, 
             'picker': graph_ode_network.picker_dim, 
             'location': graph_ode_network.location_dim}, 
            action_size, 
            graph_ode_network.hidden_dim
        )
        
        # Optimizer
        self.optimizer = optim.Adam(self.graph_network.parameters(), lr=lr)
        
        # Experience replay
        self.memory = deque(maxlen=memory_size)
        
        # Update target network
        self.update_target_network()
        
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.graph_network.state_dict())
        
    def remember(self, hetero_data, actions, rewards, next_hetero_data, dones):
        """Store experience in replay buffer"""
        self.memory.append((hetero_data, actions, rewards, next_hetero_data, dones))
        
    def act(self, hetero_data, valid_action_masks, training=True):
        """Choose actions using epsilon-greedy policy"""
        if training and np.random.random() <= self.epsilon:
            # Random action (exploration)
            actions = []
            for agent_idx in range(self.num_agents):
                valid_actions = np.where(valid_action_masks[agent_idx] == 1)[0]
                actions.append(np.random.choice(valid_actions))
            return actions
        
        # Greedy action (exploitation)
        with torch.no_grad():
            output = self.graph_network(hetero_data)
            
            actions = []
            # AGV actions
            for i in range(self.num_agvs):
                if i < output['agv_q_values'].size(0):
                    q_vals = output['agv_q_values'][i]
                    masked_q_vals = q_vals.clone()
                    masked_q_vals[valid_action_masks[i] == 0] = float('-inf')
                    actions.append(masked_q_vals.argmax().item())
                else:
                    # Fallback for missing agents
                    valid_actions = np.where(valid_action_masks[i] == 1)[0]
                    actions.append(np.random.choice(valid_actions))
                    
            # Picker actions
            for i in range(self.num_pickers):
                agent_idx = self.num_agvs + i
                if i < output['picker_q_values'].size(0):
                    q_vals = output['picker_q_values'][i]
                    masked_q_vals = q_vals.clone()
                    masked_q_vals[valid_action_masks[agent_idx] == 0] = float('-inf')
                    actions.append(masked_q_vals.argmax().item())
                else:
                    # Fallback for missing agents
                    valid_actions = np.where(valid_action_masks[agent_idx] == 1)[0]
                    actions.append(np.random.choice(valid_actions))
                
        return actions
        
    def replay(self, batch_size=32):
        """Train the network on a batch of experiences"""
        if len(self.memory) < batch_size:
            return {}
            
        batch = random.sample(self.memory, batch_size)
        
        total_loss = 0
        num_valid_samples = 0
        
        for hetero_data, actions, rewards, next_hetero_data, dones in batch:
            try:
                # Convert to tensors
                actions_tensor = torch.tensor(actions, dtype=torch.long)
                rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
                dones_tensor = torch.tensor(dones, dtype=torch.float32)
                
                # Current Q values
                current_output = self.graph_network(hetero_data)
                
                # Target Q values
                with torch.no_grad():
                    next_output = self.target_network(next_hetero_data)
                
                # Compute loss for AGVs
                if self.num_agvs > 0 and current_output['agv_q_values'].size(0) > 0:
                    agv_actions = actions_tensor[:self.num_agvs]
                    agv_rewards = rewards_tensor[:self.num_agvs]
                    agv_dones = dones_tensor[:self.num_agvs]
                    
                    # Ensure we don't exceed available agents
                    actual_agvs = min(self.num_agvs, current_output['agv_q_values'].size(0))
                    agv_actions = agv_actions[:actual_agvs]
                    agv_rewards = agv_rewards[:actual_agvs]
                    agv_dones = agv_dones[:actual_agvs]
                    
                    agv_loss = self._compute_agent_loss(
                        current_output['agv_q_values'][:actual_agvs], 
                        next_output['agv_q_values'][:actual_agvs],
                        agv_actions, agv_rewards, agv_dones
                    )
                    total_loss += agv_loss
                
                # Compute loss for Pickers
                if self.num_pickers > 0 and current_output['picker_q_values'].size(0) > 0:
                    picker_actions = actions_tensor[self.num_agvs:]
                    picker_rewards = rewards_tensor[self.num_agvs:]
                    picker_dones = dones_tensor[self.num_agvs:]
                    
                    # Ensure we don't exceed available agents
                    actual_pickers = min(self.num_pickers, current_output['picker_q_values'].size(0))
                    picker_actions = picker_actions[:actual_pickers]
                    picker_rewards = picker_rewards[:actual_pickers]
                    picker_dones = picker_dones[:actual_pickers]
                    
                    picker_loss = self._compute_agent_loss(
                        current_output['picker_q_values'][:actual_pickers],
                        next_output['picker_q_values'][:actual_pickers], 
                        picker_actions, picker_rewards, picker_dones
                    )
                    total_loss += picker_loss
                
                num_valid_samples += 1
                
            except Exception as e:
                print(f"Error in replay: {e}")
                continue
        
        if num_valid_samples > 0:
            # Backpropagation
            avg_loss = total_loss / num_valid_samples
            self.optimizer.zero_grad()
            avg_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.graph_network.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                
            return {'total_loss': avg_loss.item(), 'epsilon': self.epsilon}
        
        return {}
            
    def _compute_agent_loss(self, current_q, next_q, actions, rewards, dones):
        """Compute DQN loss for a group of agents"""
        if len(actions) == 0:
            return torch.tensor(0.0, requires_grad=True)
            
        try:
            current_q_values = current_q.gather(1, actions.unsqueeze(1))
            next_q_values = next_q.max(1)[0].detach()
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
            
            loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
            return loss
            
        except Exception as e:
            print(f"Error in agent loss computation: {e}")
            return torch.tensor(0.0, requires_grad=True)

class MultiAgentGraphConverter:
    def __init__(self, num_agvs, num_pickers, topk_tasks=5, max_comm_distance=5.0, max_task_distance=10.0):

        # Parameters for the graph observation space
        self.topk_tasks = topk_tasks
        self.max_comm_distance = max_comm_distance
        self.max_task_distance = max_task_distance

        # Node counts
        self.num_agv_nodes = num_agvs
        self.num_picker_nodes = num_pickers
        self.num_location_nodes = None

        # Node feature dimensions
        self.agv_feature_dim = 7 # [carrying_shelf, carrying_requested, toggle_loading, pos_y, pos_x, target_y, target_x]
        self.picker_feature_dim = 4 # [pos_y, pos_x, target_y, target_x]
        self.location_feature_dim = 2 # [has_shelf, is_requested]

        # Store extracted info from environment
        self._current_agents_info = []
        self._current_shelves_info = []
        self._rack_locations = []

        # Graph components
        self.node_features = None
        self.edge_index = None
        self.edge_features = None
        self.node_types = None

        # self.obs_lengths = [self.obs_length for _ in range(self.num_agents)]
        self.position_to_sections = {}

    def _build_graph_from_observation(self, observation, rack_locations):
        self.num_location_nodes = len(rack_locations)
        self._rack_locations = []
        self._rack_locations = rack_locations
        agv_features = []
        picker_features = []
        location_features = []
        self.position_to_sections = {}
        # Extract node features from current_agents_info
        agv_count= 0
        picker_count = 0

        for agent_id, obs in enumerate(observation):
            if agent_id < self.num_agv_nodes:
                agv_feature = observation[agent_id][:self.agv_feature_dim].tolist()
                self._current_agents_info.append(agv_feature)
                agv_features.append(agv_feature)
                agv_count += 1
            else:
                picker_feature = observation[agent_id][:self.picker_feature_dim].tolist()
                self._current_agents_info.append(picker_feature)
                picker_features.append(picker_feature)
                picker_count += 1
        
        # Extract task features from current_shelves_info
        shelf_data = observation[0][7+4*(self.num_agv_nodes+self.num_picker_nodes-1):]  # Start after AGV features
        for i in range(0, len(shelf_data), 2):
            location_features.append([shelf_data[i], shelf_data[i+1]])
            self._current_shelves_info.extend([shelf_data[i], shelf_data[i+1]])  # has_shelf, is_requested
        # print(f"length shelf_data: {len(shelf_data)}")
        # print(f"length self._current_shelves_info: {len(self._current_shelves_info)}")
        # build edges dynamically based on the environment
        self.edge_list = self._build_edges() #  edge_list = [agv2location_edges, location2agv_edges, agv2agv_edges, picker2location_edges, agv2picker_edges]

        for (x, y, group_idx) in self._rack_locations:
                self.position_to_sections[(x, y)] = group_idx

        data = HeteroData()
        if agv_features:
            data['agv'].num_nodes = self.num_agv_nodes
            data['agv'].x = torch.tensor(agv_features, dtype=torch.float32)
        else:
            data['agv'].num_nodes = 0
            data['agv'].x = torch.empty((0, self.agv_feature_dim), dtype=torch.float32)
        
        if picker_features:
            data['picker'].num_nodes = self.num_picker_nodes
            data['picker'].x = torch.tensor(picker_features, dtype=torch.float32)
        else:
            data['picker'].num_nodes = 0
            data['picker'].x = torch.empty((0, self.picker_feature_dim), dtype=torch.float32)

        if location_features:
            data['location'].num_nodes = self.num_location_nodes
            data['location'].x = torch.tensor(location_features, dtype=torch.float32)
        else:
            data['location'].num_nodes = 0
            data['location'].x = torch.empty((0, self.location_feature_dim), dtype=torch.float32)
        
        if self.edge_list[0]:  # agv2location_edges
            data['agv', 'targets', 'location'].edge_index = torch.tensor(self.edge_list[0], dtype=torch.long).t().contiguous()
        else:
            data['agv', 'targets', 'location'].edge_index = torch.empty((2, 0), dtype=torch.long)
        if self.edge_list[1]:  # location2agv_edges
            data['location', 'is targeted by', 'agv'].edge_index = torch.tensor(self.edge_list[1], dtype=torch.long).t().contiguous()
        else:
            data['location', 'is targeted by', 'agv'].edge_index = torch.empty((2, 0), dtype=torch.long)
        if self.edge_list[2]:  # agv2agv_edges
            data['agv', 'communicates', 'agv'].edge_index = torch.tensor(self.edge_list[2], dtype=torch.long).t().contiguous()
        else:
            data['agv', 'communicates', 'agv'].edge_index = torch.empty((2, 0), dtype=torch.long)
        if self.edge_list[3]:  # picker2location_edges
            data['picker', 'manages', 'location'].edge_index = torch.tensor(self.edge_list[3], dtype=torch.long).t().contiguous()
        else:
            data['picker', 'manages', 'location'].edge_index = torch.empty((2, 0), dtype=torch.long)
        if self.edge_list[4]:  # agv2picker_edges
            data['agv', 'cooperates with', 'picker'].edge_index = torch.tensor(self.edge_list[4], dtype=torch.long).t().contiguous()
        else:
            data['agv', 'cooperates with', 'picker'].edge_index = torch.empty((2, 0), dtype=torch.long)
        if self.edge_list[5]:  # picker2agv_edges
            data['picker', 'helps', 'agv'].edge_index = torch.tensor(self.edge_list[5], dtype=torch.long).t().contiguous()
        else:
            data['picker', 'helps', 'agv'].edge_index = torch.empty((2, 0), dtype=torch.long)

        return data
    
    def _build_edges(self):
        """
        edge_list = []

        # AGV-to-Task Edges
        agv2location_edges, location2agv_edges = self._build_agv_to_location_edges()
        # AGV-to-AGV Edges
        agv2agv_edges = self._build_agv_to_agv_edges()
        # Picker-to-Task Edges
        picker2location_edges = self._build_picker_to_location_edges()
        # AGV-to-Picker Edges
        agv2picker_edges, picker2agv_edges = self._build_agv_to_picker_edges()

        # Convert to arrays
        edge_list = [agv2location_edges, location2agv_edges, agv2agv_edges, picker2location_edges, agv2picker_edges, picker2agv_edges]
        return edge_list
        """
        # AGV-to-Task Edges
        agv2location_edges, location2agv_edges = self._build_agv_to_location_edges()
        # print(f"AGV2Location: {max([e[0] for e in agv2location_edges]) if agv2location_edges else 'empty'} -> {max([e[1] for e in agv2location_edges]) if agv2location_edges else 'empty'}")
        
        # AGV-to-AGV Edges  
        agv2agv_edges = self._build_agv_to_agv_edges()
        # print(f"AGV2AGV: {max([e[0] for e in agv2agv_edges]) if agv2agv_edges else 'empty'} -> {max([e[1] for e in agv2agv_edges]) if agv2agv_edges else 'empty'}")
        
        # Picker-to-Task Edges
        picker2location_edges = self._build_picker_to_location_edges()
        # print(f"Picker2Location: {max([e[0] for e in picker2location_edges]) if picker2location_edges else 'empty'} -> {max([e[1] for e in picker2location_edges]) if picker2location_edges else 'empty'}")
        
        # AGV-to-Picker Edges
        agv2picker_edges, picker2agv_edges = self._build_agv_to_picker_edges()
        # print(f"AGV2Picker: {max([e[0] for e in agv2picker_edges]) if agv2picker_edges else 'empty'} -> {max([e[1] for e in agv2picker_edges]) if agv2picker_edges else 'empty'}")
        
        # print(f"Current nodes - AGV: {self.num_agv_nodes}, Picker: {self.num_picker_nodes}, Location: {self.num_location_nodes}")
        edge_list = [agv2location_edges, location2agv_edges, agv2agv_edges, picker2location_edges, agv2picker_edges, picker2agv_edges]
        return edge_list       

    def _build_agv_to_location_edges(self):
        agv2location_edges = []
        location2agv_edges = []
        for agv_idx in range(self.num_agv_nodes):
            agent_info = self._current_agents_info[agv_idx]

            agv_target = np.array([agent_info[5], agent_info[6]]) # target_y, target_x
            has_target = not (agv_target[0] ==0 and agv_target[1] == 0)

            if has_target:
                for loc_idx, rack_pos in enumerate(self._rack_locations):
                    if rack_pos[0]  == agv_target[1] and rack_pos[1] == agv_target[0]:
                        agv2location_edges.extend([(agv_idx, loc_idx)])
                        location2agv_edges.extend([(loc_idx, agv_idx)])
                        break
            else:
                for location_idx, _ in enumerate(self._rack_locations):
                    location_info = self._current_shelves_info[location_idx * 2: (location_idx + 1) * 2]
                    has_shelf = location_info[0]
                    is_requested = location_info[1]
                    if has_shelf and is_requested:
                        agv2location_edges.extend([(agv_idx, location_idx)])
                        location2agv_edges.extend([(location_idx, agv_idx)])

        return agv2location_edges, location2agv_edges
    
    def _build_agv_to_agv_edges(self):
        """ Build AGV-to-AGV edges : distance  """
        agv2agv_edges = []
        for i in range(self.num_agv_nodes):
            for j in range(i+1, self.num_agv_nodes):
                agent_i_info = self._current_agents_info[i]
                agent_j_info = self._current_agents_info[j]

                pos_i = np.array([agent_i_info[4], agent_i_info[3]]) # pos_x, pos_y
                pos_j = np.array([agent_j_info[4], agent_j_info[3]]) # pos_x, pos_y

                target_i = np.array([agent_i_info[6], agent_i_info[5]]) # target_x, target_y
                target_j = np.array([agent_j_info[6], agent_j_info[5]]) # target_x, target_y

                dist = np.linalg.norm(pos_i - pos_j, ord=1)

                has_target_i = not (target_i[0] == 0 and target_i[1] == 0)
                has_target_j = not (target_j[0] == 0 and target_j[1] == 0)
                same_section = False
                if has_target_i and has_target_j:
                    same_section = self._check_same_rack_group(target_i, target_j)

                if i != j and (dist <= self.max_comm_distance or same_section):
                    agv2agv_edges.extend([(i, j), (j, i)])

        return agv2agv_edges

    def _build_picker_to_location_edges(self):
        """Build Picker-Task edges: zone-based + assigned tasks"""
        picker2location_edges = []
        for picker_idx in range(self.num_picker_nodes):
            agent_info = self._current_agents_info[self.num_agv_nodes + picker_idx]
            
            # Get picker position and target
            picker_pos = np.array([agent_info[1], agent_info[0]])  # pos_x, pos_y
            picker_target = np.array([agent_info[3], agent_info[2]])  # target_x, target_y
            has_target = not (picker_target[0] == 0 and picker_target[1] == 0)
            
            picker_section = self.position_to_sections.get((picker_pos[0], picker_pos[1]), None)

            for loc_idx, rack_pos in enumerate(self._rack_locations): # rack_pos is (x, y) format
                shelf_info = self._current_shelves_info[loc_idx * 2: (loc_idx + 1) * 2]
                loc_section = self.position_to_sections.get(rack_pos, None)
                has_shelf = shelf_info[0]
                is_requested = shelf_info[1]

                if has_target and (rack_pos == picker_target):
                    picker2location_edges.append((picker_idx, loc_idx))
                elif not has_target and picker_section == loc_section and (has_shelf and is_requested):
                    picker2location_edges.append((picker_idx, loc_idx))
        
        return picker2location_edges
    
    def _build_agv_to_picker_edges(self):
        """Build AGV-Picker edges: cooperation based"""
        agv2picker_edges = []
        picker2agv_edges = []

        for agv_idx in range(self.num_agv_nodes):
            for picker_idx in range(self.num_picker_nodes):
                agv_info = self._current_agents_info[agv_idx]
                picker_info = self._current_agents_info[self.num_agv_nodes + picker_idx]
                
                # Get positions and targets
                agv_pos = np.array([agv_info[4], agv_info[3]]) # pos_x, pos_y
                picker_pos = np.array([picker_info[1], picker_info[0]]) # pos_x, pos_y
                agv_target = np.array([agv_info[6], agv_info[5]]) # target_x, target_y
                picker_target = np.array([picker_info[3], picker_info[2]]) # target_x, target_y
                
                # Check cooperation conditions
                
                # Condition 1: Spatial proximity
                dist = np.linalg.norm(agv_pos - picker_pos, ord=1)
                close_proximity = dist <= self.max_comm_distance

                # Condition 2: Same target section coordination
                has_picker_taget = not (picker_target[0] == 0 and picker_target[1] == 0)
                has_agv_target = not (agv_target[0] == 0 and agv_target[1] == 0)
                same_target = False
                same_target_section = False
                agv_target_in_picker_section = False
                if has_agv_target and has_picker_taget:
                    same_target = picker_target[0] == agv_target[0] and picker_target[1] == agv_target[1]
                    if not same_target:
                        agv_target_section = self.position_to_sections.get((agv_target[0], agv_target[1]), None)
                        picker_target_section = self.position_to_sections.get((picker_target[0], picker_target[1]), None)
                        if agv_target_section is not None and picker_target_section is not None:
                            same_target_section = agv_target_section == picker_target_section

                else:
                    picker_current_section = self.position_to_sections.get((picker_pos[0], picker_pos[1]), None)
                    if has_agv_target:
                        agv_target_section = self.position_to_sections.get((agv_target[0], agv_target[1]), None)
                        agv_target_in_picker_section = picker_current_section == agv_target_section
                
                if close_proximity or same_target or same_target_section or agv_target_in_picker_section:
                    agv2picker_edges.append((agv_idx, picker_idx))
                    picker2agv_edges.append((picker_idx, agv_idx))
        
        return agv2picker_edges, picker2agv_edges
    
    def _check_same_rack_group(self, pos_i, pos_j):
        group_i = self.position_to_sections[(pos_i[0], pos_i[1])]
        group_j = self.position_to_sections[(pos_j[0], pos_j[1])]
        return (group_i is not None and group_j is not None and group_i == group_j)

learning_config ={'env': 'tarware-medium-19agvs-9pickers-partialobs-v1', 'ode': 'euler', 'lr': 1e-4, 'gamma': 0.999, 'epsilon_decay': 0.999, 'epsilon_min' : 0.1, 'memory_size': 100000, 'batch_size': 128, 'hidden_dim': 128}
wandb.init(
        project="swarm_ode",
        name="ode+iql",
        config=learning_config
    )
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
model_path = f'./trained_models/{current_time}'
os.makedirs(model_path, exist_ok=True)
# Initialize model
node_dims = {
    'agv': 7,  # AGV feature dimension
    'picker': 4,  # Picker feature dimension
    'location': 2  # Location feature dimension
}

env = gym.make("tarware-medium-19agvs-9pickers-partialobs-v1")
print(f"Action size: {env.unwrapped.action_size}")

num_agvs = 19
num_picker = 9

# Create Graph ODE network
graph_ode_net = HeteroGraphODENetwork(node_dims, env.unwrapped.action_size, hidden_dim=learning_config['hidden_dim'])

# # Create Modified COMA agent (with experience replay)
# agent = TypeLevelCOMA(
#     graph_ode_network=graph_ode_net,
#     num_agvs=num_agvs,
#     num_pickers=num_picker,
#     action_size=env.unwrapped.action_size,
#     lr_actor=1e-3,
#     lr_critic=1e-3,
#     gamma=0.99,
#     memory_size=50000  # Larger buffer for better sample efficiency
# )
# Create Independent DQN agent
agent = SimpleIndependentDQN(
    graph_ode_network=graph_ode_net,
    num_agvs=num_agvs,
    num_pickers=num_picker,
    action_size=env.unwrapped.action_size,
    lr=learning_config['lr'],
    gamma=learning_config['gamma'],
    epsilon=1.0,
    epsilon_decay=learning_config['epsilon_decay'],
    epsilon_min=learning_config['epsilon_min'],
    memory_size=learning_config['memory_size']
)
batch_size = learning_config['batch_size']
seed = args.seed
completed_episodes = 0
scores = deque(maxlen=100)

graph_converter = MultiAgentGraphConverter(
    num_agvs=num_agvs,
    num_pickers=num_picker,
)

for episode in range(args.num_episodes):
    state = env.reset()
    rack_locations = env.unwrapped.observation_space_mapper.get_rack_locations()
    hetero_data = graph_converter._build_graph_from_observation(state, rack_locations)

    total_reward = 0
    episode_returns = np.zeros(env.unwrapped.num_agents, dtype=np.float64)
    done = False
    step = 0
    all_infos = []
    episode_losses = []
    # episode_experiences = []
    
    # Episode execution
    while not done and step < 1000:  # Max steps per episode
        # Get valid action masks
        valid_action_masks = env.unwrapped.compute_valid_action_masks()
        
        # # Choose actions using COMA policy
        # actions, log_probs, graph_output = agent.act(hetero_data, valid_action_masks, training=True)
        
        # Choose actions using epsilon-greedy policy
        actions = agent.act(hetero_data, valid_action_masks, training=True)

        # Take environment step
        next_state, rewards, dones, truncated, info = env.step(actions)
        next_rack_locations = env.unwrapped.observation_space_mapper.get_rack_locations()
        next_hetero_data = graph_converter._build_graph_from_observation(next_state, next_rack_locations)
        
        # # COMA : Store experience for this episode
        # episode_experiences.append((hetero_data, actions, rewards, next_hetero_data, dones))
        agent.remember(hetero_data, actions, rewards, next_hetero_data, dones)

        # Update state
        hetero_data = next_hetero_data
        episode_returns += np.array(rewards, dtype=np.float64)
        total_reward += sum(rewards)
        done = all(dones)
        step += 1
        all_infos.append(info)

        if len(agent.memory) > batch_size:
            loss_info = agent.replay(batch_size)
            if loss_info:
                episode_losses.append(loss_info.get('total_loss', 0))

    if episode % 20 == 0:
        agent.update_target_network()  # Update target network every 20 episodes
        print(f"Target network updated at episode {episode}")

    # # ============ MODIFIED COMA TRAINING ============
    # # Add episode experiences to replay buffer
    # for experience in episode_experiences:
    #     agent.remember(*experience)
    
    # # Train at the end of episode using experience replay
    # losses = {}
    # if len(agent.memory) >= batch_size:
    #     # Train multiple times per episode for better learning
    #     num_updates = max(1, len(episode_experiences) // 10)  # Adaptive updates
        
    #     total_losses = {
    #         'agv_actor_loss': 0,
    #         'picker_actor_loss': 0, 
    #         'agv_critic_loss': 0,
    #         'picker_critic_loss': 0
    #     }
        
    #     for _ in range(num_updates):
    #         batch_losses = agent.replay(batch_size)
    #         for key in total_losses:
    #             total_losses[key] += batch_losses.get(key, 0)
        
    #     # Average the losses
    #     losses = {key: val / num_updates for key, val in total_losses.items()}
    
    scores.append(total_reward)
    
    # ============ LOGGING ============
    if episode % 10 == 0:
        avg_score = np.mean(scores)
        memory_size = len(agent.memory)
        print(f"Episode {episode:4d} | Avg Score: {avg_score:8.2f} | Memory: {memory_size:5d} | Epsilon: {agent.epsilon:.4f}")

        # COMA LOSS
        # if losses:
        #     print(f"             | AGV Actor: {losses['agv_actor_loss']:7.4f} | AGV Critic: {losses['agv_critic_loss']:7.4f}")
        #     print(f"             | Picker Actor: {losses['picker_actor_loss']:4.4f} | Picker Critic: {losses['picker_critic_loss']:4.4f}")
        # print("-" * 60)

        if episode % 10 == 0 and episode_losses:
            avg_loss = np.mean(episode_losses)
            print(f"Episode {episode:4d} | Avg Loss: {avg_loss:.4f}")

    # Detailed episode statistics
    last_info = info_statistics(all_infos, total_reward, episode_returns)
    last_info["overall_pick_rate"] = last_info.get("total_deliveries") * 3600 / (5 * last_info['episode_length'])
    
    print(f"Completed Episode {completed_episodes}: | [Overall Pick Rate={last_info.get('overall_pick_rate', 0):.2f}]| [Global return={last_info.get('global_episode_return', 0):.2f}]| [Total shelf deliveries={last_info.get('total_deliveries', 0):.2f}]| [Total clashes={last_info.get('total_clashes', 0):.2f}]| [Total stuck={last_info.get('total_stuck', 0):.2f}]")
    completed_episodes += 1
    wandb.log({'episode': episode, 'avg_loss': avg_loss, 'overall_pick_rate': last_info.get('overall_pick_rate'), 'global_return': last_info.get('global_episode_return'), 'total_deliveries': last_info.get('total_deliveries'), 'clashes': last_info.get('total_clashes'), 'stuck': last_info.get('total_stuck')})
    
    # Optional: Clear old experiences periodically to prevent overfitting
    if episode % 200 == 0 and episode > 0:
        print(f"Clearing replay buffer at episode {episode}")
        agent.memory.clear()

    if episode % 100 == 0 and episode > 0:
        torch.save({
            'episode': episode,
            'model_state_dict': agent.graph_network.state_dict(),
            'target_state_dict': agent.target_network.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'epsilon': agent.epsilon,
            'scores': list(scores)
        }, f'{model_path}/checkpoint_episode_{episode}.pth')
        
print("Training completed!")
print(f"Final average score: {np.mean(scores):.2f}")
print(f"Final Epsilon: {agent.epsilon:.4f}")
env.close()