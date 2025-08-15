import numpy as np
from gymnasium import spaces

from tarware.definitions import Action, AgentType, CollisionLayers
from tarware.spaces.MultiAgentBaseObservationSpace import (
    MultiAgentBaseObservationSpace, _VectorWriter)
from torch_geometric.data import Data, HeteroData
import torch

class MultiAgentGraphObservationSpace(MultiAgentBaseObservationSpace):
    def __init__(self, num_agvs, num_pickers, grid_size, shelf_locations, normalised_coordinates=False, topk_tasks=5, max_comm_distance=5.0, max_task_distance=10.0):
        super(MultiAgentGraphObservationSpace, self).__init__(num_agvs, num_pickers, grid_size, shelf_locations, normalised_coordinates)

        # Parameters for the graph observation space
        self.topk_tasks = topk_tasks
        self.max_comm_distance = max_comm_distance
        self.max_task_distance = max_task_distance
        self.type = 'graph'

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
        self._current_agents_info = []
        self._current_shelves_info = []

    def extract_environment_info(self, environment):
        self._current_agents_info = []
        self._current_shelves_info = []
        self._rack_locations = [] #(x, y)

        # Extract agents info
        for agent in environment.agents:
            agent_info = []
            if agent.type == AgentType.AGV:
                if agent.carrying_shelf:
                    agent_info.extend([1, int(agent.carrying_shelf in environment.request_queue)])
                else:
                    agent_info.extend([0, 0])
                agent_info.extend([agent.req_action == Action.TOGGLE_LOAD])
            agent_info.extend(self.process_coordinates((agent.y, agent.x), environment))
            if agent.target:
                agent_info.extend(self.process_coordinates(environment.action_id_to_coords_map[agent.target], environment))
            else:
                agent_info.extend([0, 0])
            self._current_agents_info.append(agent_info)
    
        # Extract shelves info
        for group_idx, group in enumerate(environment.rack_groups):
            for (y, x) in group:
                self._rack_locations.append((x, y))
                id_shelf = environment.grid[CollisionLayers.SHELVES, y, x]
                if id_shelf != 0:
                    self._current_shelves_info.extend([1.0, int(environment.shelfs[id_shelf - 1] in environment.request_queue), group_idx])
                else:
                    self._current_shelves_info.extend([0, 0, 0])

    
    def _build_graph_from_extracted_info(self, env):
        self.num_location_nodes = len(self._rack_locations)

        agv_features = []
        picker_features = []
        location_features = []
        self.position_to_sections = {}
        # Extract node features from current_agents_info
        agv_count= 0
        picker_count = 0

        for i, agent in enumerate(env.agents):
            agent_info = self._current_agents_info[i]

            if agent.type == AgentType.AGV:
                agv_features.append(agent_info)
                agv_count += 1

            elif agent.type == AgentType.PICKER:
                picker_features.append(agent_info)
                picker_count += 1
        
        # Extract task features from current_shelves_info
        shelf_info_idx = 0
        for i, rack_pos in enumerate(self._rack_locations): # rack_pos is (x, y) format
            has_shelf = self._current_shelves_info[shelf_info_idx]
            is_requested = self._current_shelves_info[shelf_info_idx + 1]
            group_id = self._current_shelves_info[shelf_info_idx + 2]
            features = [has_shelf, is_requested] # , group_id]
            self.position_to_sections[rack_pos] = group_id  # x, y
            location_features.append(features)
            shelf_info_idx += 3
        
        self.node_types = self._create_node_types()
        
        # build edges dynamically based on the environment
        self.edge_list = self._build_edges(env) #  edge_list = [agv2location_edges, location2agv_edges, agv2agv_edges, picker2location_edges, agv2picker_edges]


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
    def _build_edges(self, env):
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
                    location_info = self._current_shelves_info[location_idx * 3: (location_idx + 1) * 3]
                    has_shelf = location_info[0]
                    is_requested = location_info[1]
                    if has_shelf and is_requested:
                        agv2location_edges.extend([(agv_idx, loc_idx)])
                        location2agv_edges.extend([(loc_idx, agv_idx)])

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
            
            picker_section = self.position_to_sections.get(picker_pos, None)

            for loc_idx, rack_pos in enumerate(self._rack_locations): # rack_pos is (x, y) format
                shelf_info = self._current_shelves_info[loc_idx * 3: (loc_idx + 1) * 3]
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
                if has_agv_target and has_picker_taget:
                    same_target = picker_target[0] == agv_target[0] and picker_target[1] == agv_target[1]
                    if not same_target:
                        agv_target_section = self.position_to_sections.get(agv_target, None)
                        picker_target_section = self.position_to_sections.get(picker_target, None)
                        if agv_target_section is not None and picker_target_section is not None:
                            same_target_section = agv_target_section == picker_target_section

                else:
                    picker_current_section = self.position_to_sections.get(picker_pos, None)
                    if has_agv_target:
                        agv_target_section = self.position_to_sections.get(agv_target, None)
                        agv_target_in_picker_section = picker_current_section == agv_target_section
                
                if close_proximity or same_target or same_target_section or agv_target_in_picker_section:
                    agv2picker_edges.append((agv_idx, self.num_agv_nodes + picker_idx))
                    picker2agv_edges.append((self.num_agv_nodes + picker_idx, agv_idx))
        
        return agv2picker_edges, picker2agv_edges
    
    def _check_same_rack_group(self, pos_i, pos_j):
        group_i = self.position_to_sections[pos_i]
        group_j = self.position_to_sections[pos_j]
        return (group_i is not None and group_j is not None and group_i == group_j)
