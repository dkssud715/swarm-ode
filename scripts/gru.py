import gymnasium as gym
import tarware
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.data import Batch

from torchdiffeq import odeint
import wandb
import datetime
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

class HeteroGraphGRUNetwork(nn.Module):
    """Heterogeneous Graph Neural Network with GRU for MARL"""
    
    def __init__(self, node_dims, action_size, hidden_dim=256, num_layers=2, gru_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru_layers = gru_layers
        
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
        
        # GRU networks for temporal modeling
        self.agv_gru = nn.GRU(hidden_dim, hidden_dim, num_layers=gru_layers, batch_first=True)
        self.picker_gru = nn.GRU(hidden_dim, hidden_dim, num_layers=gru_layers, batch_first=True)

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
        
    def forward(self, hetero_data, agv_hidden=None, picker_hidden=None):
        """
        Args:
            hetero_data: HeteroData object
            agv_hidden: [gru_layers, n_agvs, hidden_dim] - previous hidden state for AGV GRU
            picker_hidden: [gru_layers, n_pickers, hidden_dim] - previous hidden state for Picker GRU
        """
        # 1. Initial Embeddings
        x_dict = {
            'agv': self.agv_embedding(hetero_data['agv'].x),
            'picker': self.picker_embedding(hetero_data['picker'].x),
            'location': self.location_embedding(hetero_data['location'].x)
        }
        
        # 2. GNN Processing for Collaboration Context
        for conv in self.hetero_convs:
            x_dict = conv(x_dict, hetero_data.edge_index_dict)
            x_dict = {key: torch.relu(x) for key, x in x_dict.items()}
        
        # 3. Temporal modeling with GRU
        # GRU expects input of shape [batch_size, seq_len, input_size]
        # Since we have single time step, seq_len = 1
        
        agv_embeddings = x_dict['agv']  # [n_agvs, hidden_dim]
        picker_embeddings = x_dict['picker']  # [n_pickers, hidden_dim]
        
        # Reshape for GRU: add sequence dimension
        agv_input = agv_embeddings.unsqueeze(1)  # [n_agvs, 1, hidden_dim]
        picker_input = picker_embeddings.unsqueeze(1)  # [n_pickers, 1, hidden_dim]
        
        # Apply GRU
        if agv_embeddings.size(0) > 0:  # Check if we have AGVs
            agv_output, new_agv_hidden = self.agv_gru(agv_input, agv_hidden)
            evolved_agv_embeddings = agv_output.squeeze(1)  # [n_agvs, hidden_dim]
        else:
            evolved_agv_embeddings = agv_embeddings
            new_agv_hidden = agv_hidden
            
        if picker_embeddings.size(0) > 0:  # Check if we have Pickers
            picker_output, new_picker_hidden = self.picker_gru(picker_input, picker_hidden)
            evolved_picker_embeddings = picker_output.squeeze(1)  # [n_pickers, hidden_dim]
        else:
            evolved_picker_embeddings = picker_embeddings
            new_picker_hidden = picker_hidden
        
        # Location nodes don't need temporal evolution
        final_location_embeddings = x_dict['location']
        
        # 4. Generate Action Values from Evolved States
        agv_q_values = self.agv_action_head(evolved_agv_embeddings) if evolved_agv_embeddings.size(0) > 0 else torch.empty(0, self.agv_action_head[-1].out_features)
        picker_q_values = self.picker_action_head(evolved_picker_embeddings) if evolved_picker_embeddings.size(0) > 0 else torch.empty(0, self.picker_action_head[-1].out_features)
        
        return {
            'agv_q_values': agv_q_values,
            'picker_q_values': picker_q_values,
            'agv_embeddings': evolved_agv_embeddings,
            'picker_embeddings': evolved_picker_embeddings,
            'location_embeddings': final_location_embeddings,
            'agv_hidden': new_agv_hidden,
            'picker_hidden': new_picker_hidden
        }
    
    def init_hidden(self, n_agvs, n_pickers, device):
        """Initialize hidden states for GRU"""
        agv_hidden = torch.zeros(self.gru_layers, n_agvs, self.hidden_dim, device=device) if n_agvs > 0 else None
        picker_hidden = torch.zeros(self.gru_layers, n_pickers, self.hidden_dim, device=device) if n_pickers > 0 else None
        return agv_hidden, picker_hidden

class COMAActorNetwork(nn.Module):
    """
    Individual Actor Network for each agent type
    """
    def __init__(self, input_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, obs):
        """
        Args:
            obs: [batch_size, input_dim] or [num_agents, input_dim]
        Returns:
            action_logits: [batch_size, action_dim] or [num_agents, action_dim]
        """
        return self.network(obs)
    
    def get_action_probs(self, obs, action_mask=None):
        """Get action probabilities with optional masking"""
        logits = self.forward(obs)
        
        if action_mask is not None:
            # Apply action masking by setting invalid actions to large negative value
            logits = logits.masked_fill(~action_mask, -1e9)
        
        return F.softmax(logits, dim=-1)
    
    def sample_action(self, obs, action_mask=None):
        """Sample action from policy"""
        probs = self.get_action_probs(obs, action_mask)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob


class COMACriticNetwork(nn.Module):
    """
    Centralized Critic Network
    Takes global state + all agents' actions and computes Q-value for each agent
    """
    def __init__(self, global_state_dim, n_agvs, n_pickers, action_dim, hidden_dim=128):
        super().__init__()
        self.n_agvs = n_agvs
        self.n_pickers = n_pickers
        self.n_agents = n_agvs + n_pickers
        self.action_dim = action_dim
        
        # Input: global_state + all_actions_one_hot
        input_dim = global_state_dim + (self.n_agents * action_dim)
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n_agents)  # Q-value for each agent
        )
        
    def forward(self, global_state, all_actions):
        """
        Args:
            global_state: [batch_size, global_state_dim]
            all_actions: [batch_size, n_agents] - actions of all agents
        Returns:
            q_values: [batch_size, n_agents] - Q-value for each agent
        """
        batch_size = global_state.size(0)
        
        # Convert actions to one-hot
        all_actions_one_hot = F.one_hot(all_actions.long(), num_classes=self.action_dim)
        all_actions_one_hot = all_actions_one_hot.view(batch_size, -1).float()
        
        # Concatenate global state and all actions
        critic_input = torch.cat([global_state, all_actions_one_hot], dim=1)
        
        return self.network(critic_input)


class WarehouseCOMA(nn.Module):
    """
    COMA Implementation for Warehouse Environment with Graph Neural GRU
    """
    def __init__(self, graph_network, n_agvs, n_pickers, action_dim, 
                 global_state_dim, actor_hidden_dim=64, critic_hidden_dim=128):
        super().__init__()
        
        self.graph_network = graph_network  # Your HeteroGraphGRUNetwork
        self.n_agvs = n_agvs
        self.n_pickers = n_pickers
        self.n_agents = n_agvs + n_pickers
        self.action_dim = action_dim
        
        # Get embedding dimensions from graph network
        embedding_dim = graph_network.hidden_dim
        
        # Separate actors for each agent type
        self.agv_actor = COMAActorNetwork(
            input_dim=embedding_dim,
            action_dim=action_dim,
            hidden_dim=actor_hidden_dim
        )
        
        self.picker_actor = COMAActorNetwork(
            input_dim=embedding_dim,
            action_dim=action_dim,
            hidden_dim=actor_hidden_dim
        )
        
        # Centralized critic
        self.critic = COMACriticNetwork(
            global_state_dim=global_state_dim,
            n_agvs=n_agvs,
            n_pickers=n_pickers,
            action_dim=action_dim,
            hidden_dim=critic_hidden_dim
        )
        
    def get_embeddings(self, hetero_data, agv_hidden=None, picker_hidden=None):
        """Get agent embeddings from graph network"""
        graph_output = self.graph_network(hetero_data, agv_hidden, picker_hidden)
        return graph_output
    
    def get_action_probs(self, hetero_data, action_masks=None, agv_hidden=None, picker_hidden=None):
        """
        Get action probabilities for all agents
        """
        graph_output = self.get_embeddings(hetero_data, agv_hidden, picker_hidden)
        
        agv_embeddings = graph_output['agv_embeddings']  # [n_agvs, embedding_dim]
        picker_embeddings = graph_output['picker_embeddings']  # [n_pickers, embedding_dim]
        
        # Get action masks
        agv_masks = action_masks[:self.n_agvs] if action_masks is not None else None
        picker_masks = action_masks[self.n_agvs:] if action_masks is not None else None
        
        # Get action probabilities
        agv_probs = self.agv_actor.get_action_probs(agv_embeddings, agv_masks)
        picker_probs = self.picker_actor.get_action_probs(picker_embeddings, picker_masks)
        
        return torch.cat([agv_probs, picker_probs], dim=0)
    
    def sample_actions(self, hetero_data, action_masks=None):
        """
        Sample actions for all agents
        """
        graph_output = self.get_embeddings(hetero_data)
        
        agv_embeddings = graph_output['agv_embeddings']
        picker_embeddings = graph_output['picker_embeddings']
        
        # Get action masks
        agv_masks = action_masks[:self.n_agvs] if action_masks is not None else None
        picker_masks = action_masks[self.n_agvs:] if action_masks is not None else None
        
        # Sample actions
        agv_actions, agv_log_probs = self.agv_actor.sample_action(agv_embeddings, agv_masks)
        picker_actions, picker_log_probs = self.picker_actor.sample_action(picker_embeddings, picker_masks)
        
        actions = torch.cat([agv_actions, picker_actions])
        log_probs = torch.cat([agv_log_probs, picker_log_probs])
        
        return actions, log_probs
    
    def get_critic_values(self, global_state, all_actions):
        """
        Get critic values for all agents
        """
        return self.critic(global_state, all_actions)
    
    def compute_counterfactual_advantage(self, global_state, all_actions, rewards, agent_idx):
        """
        Compute counterfactual advantage for a specific agent
        This is the key part of COMA algorithm
        """
        batch_size = global_state.size(0)
        
        # Get Q-value for current joint action
        current_q = self.critic(global_state, all_actions)[:, agent_idx]
        
        # Compute counterfactual baseline
        # Average Q-value over all possible actions for agent_idx while keeping others fixed
        counterfactual_q_values = []
        
        for action in range(self.action_dim):
            # Create counterfactual action
            counterfactual_actions = all_actions.clone()
            counterfactual_actions[:, agent_idx] = action
            
            # Get Q-value for this counterfactual action
            cf_q = self.critic(global_state, counterfactual_actions)[:, agent_idx]
            counterfactual_q_values.append(cf_q)
        
        counterfactual_q_values = torch.stack(counterfactual_q_values, dim=1)  # [batch_size, action_dim]
        
        # Get action probabilities for agent_idx
        if agent_idx < self.n_agvs:
            # AGV agent
            graph_output = self.graph_network(hetero_data)  # Need hetero_data here
            agent_embedding = graph_output['agv_embeddings'][agent_idx:agent_idx+1]
            action_probs = self.agv_actor.get_action_probs(agent_embedding)
        else:
            # Picker agent
            picker_idx = agent_idx - self.n_agvs
            graph_output = self.graph_network(hetero_data)  # Need hetero_data here
            agent_embedding = graph_output['picker_embeddings'][picker_idx:picker_idx+1]
            action_probs = self.picker_actor.get_action_probs(agent_embedding)
        
        # Compute baseline as weighted average of counterfactual Q-values
        baseline = torch.sum(action_probs * counterfactual_q_values, dim=1)
        
        # Advantage = current Q - baseline
        advantage = current_q - baseline
        
        return advantage


class COMAAgent:
    """
    COMA Agent for training
    """
    def __init__(self, coma_network, lr_actor=1e-3, lr_critic=1e-3, gamma=0.99):
        self.coma_network = coma_network
        self.gamma = gamma
        
        # Separate optimizers for actors and critic
        actor_params = list(coma_network.agv_actor.parameters()) + list(coma_network.picker_actor.parameters())
        self.actor_optimizer = torch.optim.Adam(actor_params, lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(coma_network.critic.parameters(), lr=lr_critic)
        
    def act(self, hetero_data, action_masks=None, training=True):
        """
        Select actions for all agents
        """
        if training:
            return self.coma_network.sample_actions(hetero_data, action_masks)
        else:
            # For evaluation, use greedy action selection
            action_probs = self.coma_network.get_action_probs(hetero_data, action_masks)
            actions = torch.argmax(action_probs, dim=-1)
            return actions, None
    
    def update(self, batch_data):
        """
        Update COMA networks using batch of experience (individual processing)
        """
        # Extract batch data
        hetero_data_list = batch_data['hetero_data']  # List of individual graphs
        next_hetero_data_list = batch_data['next_hetero_data']
        global_states = batch_data['global_states']
        actions = batch_data['actions']
        rewards = batch_data['rewards']
        next_global_states = batch_data['next_global_states']
        dones = batch_data['dones']
        
        batch_size = len(hetero_data_list)
        
        # Compute TD targets for critic
        with torch.no_grad():
            next_q_values = self.coma_network.get_critic_values(next_global_states, actions)
            td_targets = rewards + self.gamma * next_q_values.mean(dim=1) * (1 - dones)
        
        # Update critic
        current_q_values = self.coma_network.get_critic_values(global_states, actions)
        critic_loss = F.mse_loss(current_q_values.mean(dim=1), td_targets)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actors using policy gradient (individual processing)
        actor_loss = 0
        
        for t in range(batch_size):
            # Get embeddings for this time step
            hetero_data = hetero_data_list[t]
            graph_output = self.coma_network.get_embeddings(hetero_data)
            
            agv_embeddings = graph_output['agv_embeddings']  # [n_agvs, hidden_dim]
            picker_embeddings = graph_output['picker_embeddings']  # [n_pickers, hidden_dim]
            
            current_actions = actions[t]  # [n_agents]
            
            # Compute policy loss for AGVs
            for agv_idx in range(self.coma_network.n_agvs):
                if agv_idx < agv_embeddings.size(0):  # Check if this AGV exists
                    agv_emb = agv_embeddings[agv_idx:agv_idx+1]  # [1, hidden_dim]
                    action_probs = self.coma_network.agv_actor.get_action_probs(agv_emb)
                    taken_action = current_actions[agv_idx]
                    log_prob = torch.log(action_probs[0, taken_action] + 1e-8)
                    
                    # Simple advantage using critic values
                    with torch.no_grad():
                        advantage = current_q_values[t, agv_idx] - current_q_values[t].mean()
                    
                    actor_loss += -log_prob * advantage
            
            # Compute policy loss for Pickers
            for picker_idx in range(self.coma_network.n_pickers):
                if picker_idx < picker_embeddings.size(0):  # Check if this Picker exists
                    picker_emb = picker_embeddings[picker_idx:picker_idx+1]  # [1, hidden_dim]
                    action_probs = self.coma_network.picker_actor.get_action_probs(picker_emb)
                    taken_action = current_actions[self.coma_network.n_agvs + picker_idx]
                    log_prob = torch.log(action_probs[0, taken_action] + 1e-8)
                    
                    # Simple advantage using critic values
                    with torch.no_grad():
                        advantage = current_q_values[t, self.coma_network.n_agvs + picker_idx] - current_q_values[t].mean()
                    
                    actor_loss += -log_prob * advantage
        
        # Average the actor loss over the batch
        actor_loss = actor_loss / batch_size
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item()
        }

class SimpleIndependentDQN:
    """Simple Independent Q-Learning using Graph GRU for feature extraction"""
    
    def __init__(self, graph_gru_network, num_agvs, num_pickers, action_size, 
                 lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, 
                 epsilon_min=0.01, memory_size=10000):
        
        self.graph_network = graph_gru_network
        self.num_agvs = num_agvs
        self.num_pickers = num_pickers
        self.num_agents = num_agvs + num_pickers
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Hidden states for GRU
        self.agv_hidden = None
        self.picker_hidden = None
        
        # Target network
        self.target_network = HeteroGraphGRUNetwork(
            {'agv': graph_gru_network.agv_dim, 
             'picker': graph_gru_network.picker_dim, 
             'location': graph_gru_network.location_dim}, 
            action_size, 
            graph_gru_network.hidden_dim
        )
        
        # Optimizer
        self.optimizer = optim.Adam(self.graph_network.parameters(), lr=lr)
        
        # Experience replay
        self.memory = deque(maxlen=memory_size)
        
        # Update target network
        self.update_target_network()
        
    def reset_hidden_states(self, device):
        """Reset hidden states at the beginning of each episode"""
        self.agv_hidden, self.picker_hidden = self.graph_network.init_hidden(
            self.num_agvs, self.num_pickers, device
        )
        
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.graph_network.state_dict())
        
    def remember(self, hetero_data, actions, rewards, next_hetero_data, dones, hidden_states):
        """Store experience in replay buffer"""
        self.memory.append((hetero_data, actions, rewards, next_hetero_data, dones, hidden_states))
        
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
            output = self.graph_network(hetero_data, self.agv_hidden, self.picker_hidden)
            
            # Update hidden states
            self.agv_hidden = output['agv_hidden']
            self.picker_hidden = output['picker_hidden']
            
            actions = []
            # AGV actions
            for i in range(self.num_agvs):
                if i < output['agv_q_values'].size(0):
                    q_vals = output['agv_q_values'][i]
                    masked_q_vals = q_vals.clone()
                    masked_q_vals[valid_action_masks[i] == 0] = float('-inf')
                    actions.append(masked_q_vals.argmax().item())
                else:
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
        
        for hetero_data, actions, rewards, next_hetero_data, dones, hidden_states in batch:
            try:
                agv_hidden, picker_hidden = hidden_states
                
                # Convert to tensors
                actions_tensor = torch.tensor(actions, dtype=torch.long)
                rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
                dones_tensor = torch.tensor(dones, dtype=torch.float32)
                
                # Current Q values
                current_output = self.graph_network(hetero_data, agv_hidden, picker_hidden)
                
                # Target Q values
                with torch.no_grad():
                    next_output = self.target_network(next_hetero_data, agv_hidden, picker_hidden)
                
                # Compute loss for AGVs and Pickers (same as before)
                if self.num_agvs > 0 and current_output['agv_q_values'].size(0) > 0:
                    agv_actions = actions_tensor[:self.num_agvs]
                    agv_rewards = rewards_tensor[:self.num_agvs]
                    agv_dones = dones_tensor[:self.num_agvs]
                    
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
                
                if self.num_pickers > 0 and current_output['picker_q_values'].size(0) > 0:
                    picker_actions = actions_tensor[self.num_agvs:]
                    picker_rewards = rewards_tensor[self.num_agvs:]
                    picker_dones = dones_tensor[self.num_agvs:]
                    
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
            avg_loss = total_loss / num_valid_samples
            self.optimizer.zero_grad()
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.graph_network.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                
            return {'total_loss': avg_loss.item(), 'epsilon': self.epsilon}
        
        return {}
            
    def _compute_agent_loss(self, current_q, next_q, actions, rewards, dones):
        """Same as SimpleIndependentDQN"""
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

    def process_coordinates(self, coords):
        if self.normalised_coordinates:
            return (coords[0] / (self.grid_size[0] - 1), coords[1] / (self.grid_size[1] - 1))
        else:
            return coords

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

def extract_global_state(env, hetero_data):
    """Extract comprehensive global state for COMA critic"""
    # Get environment's internal state
    obs_mapper = env.unwrapped.observation_space_mapper
    
    global_info = []
    
    # 1. All agent states from environment
    for agv_info in obs_mapper._current_agvs_agents_info:
        global_info.extend(agv_info)
    
    for picker_info in obs_mapper._current_pickers_agents_info:
        global_info.extend(picker_info)
    
    # 2. Complete shelf information
    global_info.extend(obs_mapper._current_shelves_info)
    
    # 3. Graph structure information (dynamic edges)
    edge_counts = []
    for edge_type, edge_index in hetero_data.edge_index_dict.items():
        edge_counts.append(edge_index.size(1))  # Number of edges of each type
    global_info.extend(edge_counts)
    
    # 4. Additional global metrics
    total_requests = sum(1 for i in range(0, len(obs_mapper._current_shelves_info), 2) 
                        if obs_mapper._current_shelves_info[i+1] == 1)
    total_shelves_carried = sum(1 for i in range(0, len(obs_mapper._current_agvs_agents_info)) 
                               if len(obs_mapper._current_agvs_agents_info[i]) > 0 and obs_mapper._current_agvs_agents_info[i][0] == 1)
    
    global_info.extend([total_requests, total_shelves_carried])
    
    return torch.tensor(global_info, dtype=torch.float32)


learning_config ={'env': 'tarware-large-19agvs-9pickers-partialobs-v1', 'ode': 'euler', 'lr': 1e-4, 'gamma': 0.999, 'epsilon_decay': 0.999, 'epsilon_min' : 0.1, 'memory_size': 100000, 'batch_size': 128, 'hidden_dim': 128}
wandb.init(
        project="swarm_ode",
        name="gru+iql",
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

env = gym.make("tarware-large-19agvs-9pickers-partialobs-v1")
print(f"Action size: {env.unwrapped.action_size}")
action_dim = env.unwrapped.action_size  # Action dimension
num_agvs = 19
num_picker = 9

# Initialize graph converter
graph_converter = MultiAgentGraphConverter(
    num_agvs=num_agvs,
    num_pickers=num_picker,
)

graph_network = HeteroGraphGRUNetwork(
    node_dims=node_dims,
    action_size=action_dim,
    hidden_dim=learning_config['hidden_dim'],
    num_layers=2
)

# Global state dimension calculation
# sample_state = env.reset()
# sample_rack_locations = env.unwrapped.observation_space_mapper.get_rack_locations()
# sample_hetero_data = graph_converter._build_graph_from_observation(sample_state, sample_rack_locations)
# sample_global_state = extract_global_state(env, sample_hetero_data)
# global_state_dim = sample_global_state.size(0)
# print(f"Global state dimension: {global_state_dim}")

# coma_network = WarehouseCOMA(
#     graph_network=graph_network,
#     n_agvs=num_agvs,
#     n_pickers=num_picker,
#     action_dim=action_dim,
#     global_state_dim=global_state_dim
# )

# agent = COMAAgent(coma_network, lr_actor=1e-4, lr_critic=1e-3)
agent = SimpleIndependentDQN(
    graph_gru_network=graph_network,
    num_agvs=num_agvs,
    num_pickers=num_picker,
    action_size=env.unwrapped.action_size,
    lr=learning_config['lr'],
    gamma=learning_config['gamma'],
    epsilon=learning_config['epsilon'],
    epsilon_decay=learning_config['epsilon_decay'],
    epsilon_min=learning_config['epsilon_min'],
    memory_size=learning_config['memory_size']
)
# Training parameters
#  update_frequency = 4  # Update every N episodes for COMA
batch_size = learning_config['batch_size']
seed = args.seed
completed_episodes = 0
scores = deque(maxlen=100)

# # Episode buffer for COMA (collect full episodes before training)
# episode_buffer = {
#     'hetero_data': [],
#     'global_states': [],
#     'actions': [],
#     'rewards': [],
#     'next_hetero_data': [],
#     'next_global_states': [],
#     'dones': [],
#     'agv_hiddens': [],
#     'picker_hiddens': []
# }

# def clear_episode_buffer():
#     """Clear episode buffer"""
#     for key in episode_buffer.keys():
#         episode_buffer[key] = []

for episode in range(args.num_episodes):
    state = env.reset()
    rack_locations = env.unwrapped.observation_space_mapper.get_rack_locations()
    hetero_data = graph_converter._build_graph_from_observation(state, rack_locations)
    
    # Initialize hidden states for this episode
    agv_hidden, picker_hidden = graph_network.init_hidden(num_agvs, num_picker, device=next(graph_network.parameters()).device)
    agent.reset_hidden_states(device=next(graph_network.parameters()).device)

    total_reward = 0
    episode_returns = np.zeros(env.unwrapped.num_agents, dtype=np.float64)
    done = False
    step = 0
    all_infos = []
    episode_losses = []
    # # Clear episode buffer
    # clear_episode_buffer()
    
    while not done and step < 1000:  # Max steps per episode
        # Get valid action masks
        valid_action_masks = env.unwrapped.compute_valid_action_masks()
        valid_action_masks = torch.tensor(valid_action_masks, dtype=torch.bool)
        
        # # Choose actions (with hidden states)
        # graph_output = graph_network(hetero_data, agv_hidden, picker_hidden)
        
        # # Sample actions using COMA actors
        # agv_embeddings = graph_output['agv_embeddings']
        # picker_embeddings = graph_output['picker_embeddings']
        
        # # Get action masks
        # agv_masks = valid_action_masks[:num_agvs]
        # picker_masks = valid_action_masks[num_agvs:]
        
        # # Sample actions
        # agv_actions, agv_log_probs = agent.coma_network.agv_actor.sample_action(agv_embeddings, agv_masks)
        # picker_actions, picker_log_probs = agent.coma_network.picker_actor.sample_action(picker_embeddings, picker_masks)
        
        # actions = torch.cat([agv_actions, picker_actions]).cpu().numpy()
        
        # # Update hidden states
        # agv_hidden = graph_output['agv_hidden']
        # picker_hidden = graph_output['picker_hidden']
        
        # # Take environment step
        # next_state, rewards, dones, truncated, info = env.step(actions)
        # next_rack_locations = env.unwrapped.observation_space_mapper.get_rack_locations()
        # next_hetero_data = graph_converter._build_graph_from_observation(next_state, next_rack_locations)
        
        # # Extract global states
        # global_state = extract_global_state(env, hetero_data)
        # next_global_state = extract_global_state(env, next_hetero_data)
        
        # # Store experience in episode buffer
        # episode_buffer['hetero_data'].append(hetero_data)
        # episode_buffer['global_states'].append(global_state)
        # episode_buffer['actions'].append(torch.tensor(actions))
        # episode_buffer['rewards'].append(torch.tensor(sum(rewards), dtype=torch.float32))  # Convert to scalar
        # episode_buffer['next_hetero_data'].append(next_hetero_data)
        # episode_buffer['next_global_states'].append(next_global_state)
        # episode_buffer['dones'].append(torch.tensor(float(all(dones)), dtype=torch.float32))  # Convert to scalar
        # episode_buffer['agv_hiddens'].append(agv_hidden.clone() if agv_hidden is not None else None)
        # episode_buffer['picker_hiddens'].append(picker_hidden.clone() if picker_hidden is not None else None)

        # Choose actions using epsilon-greedy policy
        actions = agent.act(hetero_data, valid_action_masks, training=True)

        # Take environment step
        next_state, rewards, dones, truncated, info = env.step(actions)
        next_rack_locations = env.unwrapped.observation_space_mapper.get_rack_locations()
        next_hetero_data = graph_converter._build_graph_from_observation(next_state, next_rack_locations)

        # Store experience in replay buffer (hidden states 포함)
        hidden_states = (agent.agv_hidden.clone() if agent.agv_hidden is not None else None,
                        agent.picker_hidden.clone() if agent.picker_hidden is not None else None)
        agent.remember(hetero_data, actions, rewards, next_hetero_data, dones, hidden_states)

        # Update state
        hetero_data = next_hetero_data
        state = next_state
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
    scores.append(total_reward)
    
    # # Train COMA after collecting full episode
    # if len(episode_buffer['hetero_data']) > 0:
    #     # Prepare batch data (no batching for graph data)
    #     batch_data = {
    #         'hetero_data': episode_buffer['hetero_data'],  # List of individual graphs
    #         'next_hetero_data': episode_buffer['next_hetero_data'],
    #         'global_states': torch.stack(episode_buffer['global_states']),
    #         'actions': torch.stack(episode_buffer['actions']),
    #         'rewards': torch.stack(episode_buffer['rewards']),
    #         'next_global_states': torch.stack(episode_buffer['next_global_states']),
    #         'dones': torch.stack(episode_buffer['dones']),
    #         'agv_hiddens': episode_buffer['agv_hiddens'],
    #         'picker_hiddens': episode_buffer['picker_hiddens']
    #     }
        
        # # Update COMA (you might want to do this every few episodes to accumulate more data)
        # if episode % update_frequency == 0:
        #     loss_info = agent.update(batch_data)
        #     if episode % 10 == 0:
        #         print(f"Training - Critic Loss: {loss_info['critic_loss']:.4f}, Actor Loss: {loss_info['actor_loss']:.4f}")
    
    # ============ LOGGING ============
    if episode % 10 == 0:
        avg_score = np.mean(scores)
        memory_size = len(agent.memory)
        print(f"Episode {episode:4d} | Avg Score: {avg_score:8.2f} | Memory: {memory_size:5d} | Epsilon: {agent.epsilon:.4f}")
        if episode_losses :
            avg_loss = np.mean(episode_losses)
            print(f"Episode {episode:4d} | Avg Loss: {avg_loss:.4f}")
    # Episode statistics
    last_info = info_statistics(all_infos, total_reward, episode_returns)
    last_info["overall_pick_rate"] = last_info.get("total_deliveries", 0) * 3600 / (5 * last_info.get('episode_length', 1))
    
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