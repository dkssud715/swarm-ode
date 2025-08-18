import gymnasium as gym
import tarware
import torch
import torch.nn as nn
import torch.optim as optim

from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv

from torchdiffeq import odeint

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
    """Heterogeneous Graph Neural ODE for MARL"""
    
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
        
        # ODE Function Network
        self.ode_func = ODEFunction(hidden_dim, ode_hidden_dim)
        
        # Action heads for each agent type
        self.agv_action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_size)  # Q-value for each possible action
        )
        
        self.picker_action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_size)  # Q-value for each possible action
        )
        
    def forward(self, hetero_data, integration_time=1.0):
        # Initial embeddings
        x_dict = {
            'agv': self.agv_embedding(hetero_data['agv'].x),
            'picker': self.picker_embedding(hetero_data['picker'].x),
            'location': self.location_embedding(hetero_data['location'].x)
        }
        
        # Initial GNN processing
        for conv in self.hetero_convs:
            x_dict = conv(x_dict, hetero_data.edge_index_dict)
            x_dict = {key: torch.relu(x) for key, x in x_dict.items()}
        
        # Concatenate all node embeddings for ODE
        all_embeddings = torch.cat([
            x_dict['agv'], x_dict['picker'], x_dict['location']
        ], dim=0)
        
        # Apply Neural ODE
        t = torch.tensor([0., integration_time], dtype=torch.float32)
        evolved_embeddings = odeint(self.ode_func, all_embeddings, t)[-1]
        
        # Split back to node types
        num_agv = hetero_data['agv'].num_nodes
        num_picker = hetero_data['picker'].num_nodes
        num_location = hetero_data['location'].num_nodes
        
        agv_embeddings = evolved_embeddings[:num_agv]
        picker_embeddings = evolved_embeddings[num_agv:num_agv + num_picker]
        location_embeddings = evolved_embeddings[num_agv + num_picker:]
        
        # Generate action values
        agv_q_values = self.agv_action_head(agv_embeddings)
        picker_q_values = self.picker_action_head(picker_embeddings)
        
        return {
            'agv_q_values': agv_q_values,
            'picker_q_values': picker_q_values,
            'agv_embeddings': agv_embeddings,
            'picker_embeddings': picker_embeddings,
            'location_embeddings': location_embeddings
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

class GraphMARL_DQN:
    """Graph-based Multi-Agent Deep Q-Network with Neural ODE"""
    
    def __init__(self, node_dims, action_size, lr=1e-3, gamma=0.99, epsilon=1.0, 
                 epsilon_decay=0.995, epsilon_min=0.01, memory_size=10000):
        
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Neural networks
        self.q_network = HeteroGraphODENetwork(node_dims, action_size)
        self.target_network = HeteroGraphODENetwork(node_dims, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Experience replay
        self.memory = deque(maxlen=memory_size)
        
        # Update target network
        self.update_target_network()
        
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def remember(self, state, actions, rewards, next_state, dones):
        """Store experience in replay buffer"""
        self.memory.append((state, actions, rewards, next_state, dones))
        
    def act(self, hetero_data, valid_action_masks, training=True):
        """Choose actions using epsilon-greedy policy"""
        if training and np.random.random() <= self.epsilon:
            # Random action (exploration)
            actions = []
            for agent_idx in range(hetero_data['agv'].num_nodes + hetero_data['picker'].num_nodes):
                if agent_idx < hetero_data['agv'].num_nodes:
                    # AGV action
                    valid_actions = np.where(valid_action_masks[agent_idx] == 1)[0]
                else:
                    # Picker action  
                    valid_actions = np.where(valid_action_masks[agent_idx] == 1)[0]
                actions.append(np.random.choice(valid_actions))
            return actions
        
        # Greedy action (exploitation)
        with torch.no_grad():
            output = self.q_network(hetero_data)
            
            actions = []
            # AGV actions
            for i, q_vals in enumerate(output['agv_q_values']):
                masked_q_vals = q_vals.clone()
                masked_q_vals[valid_action_masks[i] == 0] = float('-inf')
                actions.append(masked_q_vals.argmax().item())
                
            # Picker actions
            agv_count = hetero_data['agv'].num_nodes
            for i, q_vals in enumerate(output['picker_q_values']):
                agent_idx = agv_count + i
                masked_q_vals = q_vals.clone()
                masked_q_vals[valid_action_masks[agent_idx] == 0] = float('-inf')
                actions.append(masked_q_vals.argmax().item())
                
        return actions
        
    def replay(self, batch_size=32):
        """Train the network on a batch of experiences"""
        if len(self.memory) < batch_size:
            return
            
        batch = random.sample(self.memory, batch_size)
        
        total_loss = 0
        for state, actions, rewards, next_state, dones in batch:
            # Current Q values
            # print(f"State node counts: {[x.size(0) if hasattr(x, 'size') else len(x) for x in state.x_dict.values()]}")
            # print(f"Max edge indices: {[edge_idx.max().item() for edge_idx in state.edge_index_dict.values()]}")

            current_output = self.q_network(state)
            
            # Target Q values
            with torch.no_grad():
                next_output = self.target_network(next_state)
                
            # Compute loss for each agent type
            agv_loss = self._compute_agent_loss(
                current_output['agv_q_values'], 
                next_output['agv_q_values'],
                actions[:state['agv'].num_nodes],
                rewards[:state['agv'].num_nodes],
                dones[:state['agv'].num_nodes]
            )
            
            picker_loss = self._compute_agent_loss(
                current_output['picker_q_values'],
                next_output['picker_q_values'], 
                actions[state['agv'].num_nodes:],
                rewards[state['agv'].num_nodes:],
                dones[state['agv'].num_nodes:]
            )
            
            loss = agv_loss + picker_loss
            total_loss += loss
            
        # Backpropagation
        self.optimizer.zero_grad()
        (total_loss / batch_size).backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def _compute_agent_loss(self, current_q, next_q, actions, rewards, dones):
        """Compute DQN loss for a group of agents"""                         
        current_q_values = current_q.gather(1, torch.tensor(actions).unsqueeze(1))
        next_q_values = next_q.max(1)[0].detach()

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        dones_tensor = torch.tensor(dones, dtype=torch.float32)
        target_q_values = rewards_tensor + (self.gamma * next_q_values * (1 - dones_tensor))
        
        return nn.MSELoss()(current_q_values.squeeze(), target_q_values)

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

# Initialize model
node_dims = {
    'agv': 7,  # AGV feature dimension
    'picker': 4,  # Picker feature dimension
    'location': 2  # Location feature dimension
}
env = gym.make("tarware-large-19agvs-9pickers-partialobs-v1")
print(env.unwrapped.action_size)
agent = GraphMARL_DQN(
    node_dims=node_dims,
    action_size=env.unwrapped.action_size,
    lr=1e-3,
    gamma=0.99
)
batch_size = 64
seed = args.seed
completed_episodes = 0
scores = deque(maxlen=100)
num_agvs = 19
num_picker = 9
grid_size = (35, 22)
graph_converter = MultiAgentGraphConverter(
    num_agvs = num_agvs,
    num_pickers = num_picker,
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
    while not done and step < 1000:  # Max steps per episode
        # Get valid action masks
        valid_action_masks = env.unwrapped.compute_valid_action_masks()
        
        # Choose actions
        actions = agent.act(hetero_data, valid_action_masks)
        
        # Take environment step
        next_state, rewards, dones, truncated, info = env.step(actions)
        next_rack_locations = env.unwrapped.observation_space_mapper.get_rack_locations()
        next_hetero_data = graph_converter._build_graph_from_observation(next_state, next_rack_locations)
        
        # Store experience
        agent.remember(hetero_data, actions, rewards, next_hetero_data, dones)
        
        # Update state
        hetero_data = next_hetero_data
        episode_returns += np.array(rewards, dtype=np.float64)
        total_reward += sum(rewards)
        done = all(dones)
        step += 1
        all_infos.append(info)
        # Train the model
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
            
    # Update target network periodically
    if episode % 100 == 0:
        agent.update_target_network()
        
    scores.append(total_reward)
    
    # Logging
    if episode % 100 == 0:
        avg_score = np.mean(scores)
        print(f"Episode {episode}, Average Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}")
    last_info = info_statistics(all_infos, total_reward, episode_returns)
    last_info["overall_pick_rate"] = last_info.get("total_deliveries") * 3600 / (5 * last_info['episode_length'])
    episode_length = len(info)
    print(f"Completed Episode {completed_episodes}: | [Overall Pick Rate={last_info.get('overall_pick_rate'):.2f}]| [Global return={last_info.get('global_episode_return'):.2f}]| [Total shelf deliveries={last_info.get('total_deliveries'):.2f}]| [Total clashes={last_info.get('total_clashes'):.2f}]| [Total stuck={last_info.get('total_stuck'):.2f}]")
    completed_episodes += 1

env.close()