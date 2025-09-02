import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, GATConv, Linear
from torch_geometric.data import HeteroData, Batch
# Import the converter from your code
from torchdiffeq import odeint
import h5py
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt
from collections import defaultdict

@dataclass
class TrajectoryBatch:
    """Batch of trajectory data for training"""
    hetero_data: HeteroData  # Graph data
    trajectories: Dict[str, torch.Tensor]  # Node type -> [B, T, N, F] trajectories
    time_steps: torch.Tensor  # [T] time points
    
class WarehouseDataLoader:
    """Loads and preprocesses warehouse trajectory data"""
    
    def __init__(self, h5_file_path: str, sequence_length: int = 10, prediction_horizon: int = 5):
        self.h5_file_path = h5_file_path
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.converter = None
        
    def load_episode_data(self, episode_id: int) -> List[Dict]:
        """Load single episode data"""
        with h5py.File(self.h5_file_path, 'r') as f:
            episode_group = f[f'episode_{episode_id:06d}']
            steps_group = episode_group['steps']
            
            episode_data = []
            rack_locations = episode_group['metadata/rack_locations'][:]
            num_agvs = episode_group['metadata'].attrs['num_agvs']
            num_pickers = episode_group['metadata'].attrs['num_pickers']
            
            # Initialize converter if not exists
            if self.converter is None:
                self.converter = MultiAgentGraphConverter(
                    num_agvs=num_agvs,
                    num_pickers=num_pickers,
                    topk_tasks=5,
                    max_comm_distance=5.0,
                    max_task_distance=10.0
                )
            
            for step_name in sorted(steps_group.keys()):
                step_data = steps_group[step_name]
                
                # Extract observations - this contains the graph features
                if 'observations' in step_data:
                    observations = step_data['observations'][:]
                    # Convert to graph using your converter
                    graph_data = self.converter._build_graph_from_observation(
                        observations, rack_locations
                    )
                    episode_data.append({
                        'graph': graph_data,
                        'raw_obs': observations,
                        'step_id': int(step_name.split('_')[1])
                    })
                    
            return episode_data
    
    def create_trajectory_sequences(self, episode_data: List[Dict]) -> List[TrajectoryBatch]:
        """Create sequences for trajectory prediction"""
        sequences = []
        
        for i in range(len(episode_data) - self.sequence_length - self.prediction_horizon + 1):
            # Input sequence
            input_graphs = [episode_data[i + j]['graph'] for j in range(self.sequence_length)]
            
            # Target sequence  
            target_graphs = [episode_data[i + self.sequence_length + j]['graph'] 
                           for j in range(self.prediction_horizon)]
            
            # Extract trajectories from graph features
            trajectories = self._extract_trajectories(input_graphs + target_graphs)
            
            # Use the last input graph as the graph structure
            hetero_data = input_graphs[-1]
            
            sequences.append(TrajectoryBatch(
                hetero_data=hetero_data,
                trajectories=trajectories,
                time_steps=torch.linspace(0, 1, self.sequence_length + self.prediction_horizon)
            ))
            
        return sequences
    
    def _extract_trajectories(self, graphs: List[HeteroData]) -> Dict[str, torch.Tensor]:
        """Extract position trajectories from graph sequence"""
        trajectories = {}
        
        for node_type in ['agv', 'picker']:
            if node_type in graphs[0]:
                node_trajs = []
                for graph in graphs:
                    if node_type == 'agv':
                        # AGV features: [carrying_shelf, carrying_requested, toggle_loading, pos_y, pos_x, target_y, target_x]
                        positions = graph[node_type].x[:, [4, 3]]  # [pos_x, pos_y]
                    else:  # picker
                        # Picker features: [pos_y, pos_x, target_y, target_x]
                        positions = graph[node_type].x[:, [1, 0]]  # [pos_x, pos_y]
                    node_trajs.append(positions)
                
                # Stack: [T, N, 2]
                trajectories[node_type] = torch.stack(node_trajs, dim=0)
        
        return trajectories

class HeteroGraphODEFunc(nn.Module):
    """Heterogeneous Graph ODE Function"""
    
    def __init__(self, 
                 node_dims: Dict[str, int],
                 edge_types: List[Tuple[str, str, str]],
                 hidden_dim: int = 64):
        super().__init__()
        self.node_dims = node_dims
        self.edge_types = edge_types
        self.hidden_dim = hidden_dim
        
        # Heterogeneous graph convolution layers
        self.conv1 = HeteroConv({
            edge_type: GCNConv(-1, hidden_dim) 
            for edge_type in edge_types
        }, aggr='sum')
        
        self.conv2 = HeteroConv({
            edge_type: GCNConv(hidden_dim, hidden_dim) 
            for edge_type in edge_types  
        }, aggr='sum')
        
        # Output projections for each node type
        self.output_projs = nn.ModuleDict({
            node_type: Linear(hidden_dim, 2)  # 2D position derivatives
            for node_type in node_dims.keys()
        })
        
    def forward(self, t, x_dict):
        """
        Args:
            t: time (not used in autonomous system)
            x_dict: Dict[str, Tensor] - node features by type
        Returns:
            dx_dict: Dict[str, Tensor] - derivatives by node type
        """
        # Apply graph convolutions
        h_dict = self.conv1(x_dict, self.edge_index_dict)
        h_dict = {key: F.relu(h) for key, h in h_dict.items()}
        h_dict = self.conv2(h_dict, self.edge_index_dict)
        
        # Compute derivatives for each node type
        dx_dict = {}
        for node_type, h in h_dict.items():
            if node_type in self.output_projs:
                dx_dict[node_type] = self.output_projs[node_type](h)
                
        return dx_dict
    
    def set_graph_structure(self, edge_index_dict):
        """Set the current graph structure"""
        self.edge_index_dict = edge_index_dict

class WarehouseGraphODE(nn.Module):
    """Complete Warehouse Graph ODE Model"""
    
    def __init__(self, 
                 node_dims: Dict[str, int],
                 edge_types: List[Tuple[str, str, str]],
                 hidden_dim: int = 64,
                 method: str = 'dopri5'):
        super().__init__()
        
        self.node_dims = node_dims
        self.hidden_dim = hidden_dim
        self.method = method
        
        # Feature encoders for each node type
        self.encoders = nn.ModuleDict({
            'agv': nn.Sequential(
                Linear(7, hidden_dim),  # AGV features
                nn.ReLU(),
                Linear(hidden_dim, hidden_dim)
            ),
            'picker': nn.Sequential(
                Linear(4, hidden_dim),  # Picker features  
                nn.ReLU(),
                Linear(hidden_dim, hidden_dim)
            ),
            'location': nn.Sequential(
                Linear(2, hidden_dim),  # Location features
                nn.ReLU(), 
                Linear(hidden_dim, hidden_dim)
            )
        })
        
        # ODE function
        self.ode_func = HeteroGraphODEFunc(node_dims, edge_types, hidden_dim)
        
        # Position extractors (for positions from encoded features)
        self.position_extractors = nn.ModuleDict({
            'agv': Linear(hidden_dim, 2),
            'picker': Linear(hidden_dim, 2)
        })
        
    def forward(self, hetero_data: HeteroData, time_steps: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Graph ODE
        
        Args:
            hetero_data: Heterogeneous graph data
            time_steps: [T] time points to solve for
            
        Returns:
            trajectories: Dict[str, Tensor] - predicted trajectories [T, N, 2]
        """
        # Set graph structure in ODE function
        edge_index_dict = {}
        for edge_type in hetero_data.edge_types:
            edge_index_dict[edge_type] = hetero_data[edge_type].edge_index
        self.ode_func.set_graph_structure(edge_index_dict)
        
        # Encode initial node features
        x0_dict = {}
        for node_type in hetero_data.node_types:
            if node_type in self.encoders and hetero_data[node_type].num_nodes > 0:
                x0_dict[node_type] = self.encoders[node_type](hetero_data[node_type].x)
        
        # Solve ODE
        trajectory_dict = odeint(
            self.ode_func, 
            x0_dict, 
            time_steps,
            method=self.method
        )
        
        # Extract positions from trajectories
        position_trajectories = {}
        for node_type in ['agv', 'picker']:
            if node_type in trajectory_dict:
                # trajectory_dict[node_type]: [T, N, hidden_dim]
                positions = self.position_extractors[node_type](trajectory_dict[node_type])
                position_trajectories[node_type] = positions  # [T, N, 2]
        
        return position_trajectories

class WarehouseTrajectoryPredictor:
    """Complete trajectory prediction system"""
    
    def __init__(self, h5_file_path: str, hidden_dim: int = 64):
        self.data_loader = WarehouseDataLoader(h5_file_path)
        self.hidden_dim = hidden_dim
        self.model = None
        
    def initialize_model(self, sample_data: HeteroData):
        """Initialize model based on data structure"""
        # Extract node dimensions and edge types from sample data
        node_dims = {}
        for node_type in sample_data.node_types:
            if sample_data[node_type].num_nodes > 0:
                node_dims[node_type] = sample_data[node_type].x.shape[1]
        
        edge_types = list(sample_data.edge_types)
        
        self.model = WarehouseGraphODE(
            node_dims=node_dims,
            edge_types=edge_types, 
            hidden_dim=self.hidden_dim
        )
        
        return self.model
    
    def train_episode(self, episode_id: int, num_epochs: int = 100):
        """Train on single episode"""
        # Load episode data
        episode_data = self.data_loader.load_episode_data(episode_id)
        sequences = self.data_loader.create_trajectory_sequences(episode_data)
        
        if not sequences:
            print("No sequences found!")
            return
            
        # Initialize model if needed
        if self.model is None:
            self.initialize_model(sequences[0].hetero_data)
            
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        
        print(f"Training on {len(sequences)} sequences...")
        
        for epoch in range(num_epochs):
            total_loss = 0
            
            for seq_batch in sequences:
                optimizer.zero_grad()
                
                # Predict trajectories
                pred_trajectories = self.model(
                    seq_batch.hetero_data,
                    seq_batch.time_steps
                )
                
                # Compute loss (MSE on positions)
                loss = 0
                for node_type in pred_trajectories:
                    if node_type in seq_batch.trajectories:
                        pred = pred_trajectories[node_type]  # [T, N, 2]
                        target = seq_batch.trajectories[node_type]  # [T, N, 2]
                        
                        # Only predict future steps
                        pred_future = pred[self.data_loader.sequence_length:]
                        target_future = target[self.data_loader.sequence_length:]
                        
                        loss += F.mse_loss(pred_future, target_future)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            if epoch % 10 == 0:
                avg_loss = total_loss / len(sequences)
                print(f"Epoch {epoch}: Loss = {avg_loss:.6f}")
    
    def predict_trajectory(self, hetero_data: HeteroData, num_steps: int = 10) -> Dict[str, torch.Tensor]:
        """Predict future trajectories"""
        if self.model is None:
            raise ValueError("Model not initialized! Train first.")
            
        time_steps = torch.linspace(0, 1, num_steps)
        
        with torch.no_grad():
            trajectories = self.model(hetero_data, time_steps)
            
        return trajectories
    
    def visualize_predictions(self, hetero_data: HeteroData, trajectories: Dict[str, torch.Tensor]):
        """Visualize predicted trajectories"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Plot AGV trajectories
        if 'agv' in trajectories:
            agv_traj = trajectories['agv'].numpy()  # [T, N, 2]
            for agent_idx in range(agv_traj.shape[1]):
                ax.plot(agv_traj[:, agent_idx, 0], agv_traj[:, agent_idx, 1], 
                       'b-', alpha=0.7, label=f'AGV {agent_idx}' if agent_idx == 0 else '')
                ax.scatter(agv_traj[0, agent_idx, 0], agv_traj[0, agent_idx, 1], 
                          c='blue', s=100, marker='s')
                ax.scatter(agv_traj[-1, agent_idx, 0], agv_traj[-1, agent_idx, 1], 
                          c='blue', s=100, marker='*')
        
        # Plot Picker trajectories  
        if 'picker' in trajectories:
            picker_traj = trajectories['picker'].numpy()  # [T, N, 2]
            for agent_idx in range(picker_traj.shape[1]):
                ax.plot(picker_traj[:, agent_idx, 0], picker_traj[:, agent_idx, 1], 
                       'r-', alpha=0.7, label=f'Picker {agent_idx}' if agent_idx == 0 else '')
                ax.scatter(picker_traj[0, agent_idx, 0], picker_traj[0, agent_idx, 1], 
                          c='red', s=100, marker='s')
                ax.scatter(picker_traj[-1, agent_idx, 0], picker_traj[-1, agent_idx, 1], 
                          c='red', s=100, marker='*')
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position') 
        ax.set_title('Warehouse Agent Trajectory Predictions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

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

        self.position_to_sections = {}

    def _build_graph_from_observation(self, observation, rack_locations):
        self.num_location_nodes = len(rack_locations)
        self._rack_locations = []
        self._rack_locations = rack_locations
        agv_features = []
        picker_features = []
        location_features = []
        self.position_to_sections = {}
        # Reset agent and shelf info for this step
        self._current_agents_info = []
        self._current_shelves_info = []
        
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
        # AGV-to-Task Edges
        agv2location_edges, location2agv_edges = self._build_agv_to_location_edges()
        # AGV-to-AGV Edges  
        agv2agv_edges = self._build_agv_to_agv_edges()
        # Picker-to-Task Edges
        picker2location_edges = self._build_picker_to_location_edges()
        # AGV-to-Picker Edges
        agv2picker_edges, picker2agv_edges = self._build_agv_to_picker_edges()
        
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
        group_i = self.position_to_sections.get((pos_i[0], pos_i[1]), None)
        group_j = self.position_to_sections.get((pos_j[0], pos_j[1]), None)
        return (group_i is not None and group_j is not None and group_i == group_j)

# Usage Example
if __name__ == "__main__":
    # Initialize predictor
    # environments = [
    #     'tarware-tiny-3agvs-2pickers-partialobs-v1',
    #     'tarware-small-6agvs-3pickers-partialobs-v1', 
    #     'tarware-medium-10agvs-5pickers-partialobs-v1',
    #     'tarware-medium-19agvs-9pickers-partialobs-v1',
    #     'tarware-large-15agvs-8pickers-partialobs-v1'
    # ]
    h5_file_path = "tarware-tiny-3agvs-2pickers-partialobs-v1_seed0.h5"
    predictor = WarehouseTrajectoryPredictor(h5_file_path, hidden_dim=64)
    
    # Train on first episode
    print("Training Graph ODE model...")
    predictor.train_episode(episode_id=0, num_epochs=100)
    
    # Load test data
    test_episode_data = predictor.data_loader.load_episode_data(episode_id=1)
    if test_episode_data:
        test_graph = test_episode_data[10]['graph']  # Use step 10 as initial condition
        
        # Predict future trajectories
        print("Predicting trajectories...")
        predicted_trajectories = predictor.predict_trajectory(test_graph, num_steps=15)
        
        # Visualize
        print("Visualizing predictions...")
        predictor.visualize_predictions(test_graph, predicted_trajectories)
        
        # Print trajectory statistics
        for node_type, traj in predicted_trajectories.items():
            print(f"{node_type} trajectory shape: {traj.shape}")
            print(f"{node_type} position range: x[{traj[:,:,0].min():.2f}, {traj[:,:,0].max():.2f}], y[{traj[:,:,1].min():.2f}, {traj[:,:,1].max():.2f}]")