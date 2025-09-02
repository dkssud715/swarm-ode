import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import gymnasium as gym
import numpy as np
from collections import defaultdict, OrderedDict

from tarware.heuristic import heuristic_episode
from tarware.definitions import AgentType
from tarware.utils.utils import flatten_list, split_list
import h5py

class HDF5Logger:
    def __init__(self, filepath):
        self.filepath = filepath
        self.file = None
        self.current_episode_group = None
        self.step_data_buffer = []
        
    def start_episode(self, episode_id, seed, env):
        if self.file is None:
            self.file = h5py.File(self.filepath, 'w')
            
        # Episode 메타데이터 저장
        ep_group = self.file.create_group(f'episode_{episode_id:06d}')
        self.current_episode_group = ep_group
        
        # 환경 정보 저장
        meta_group = ep_group.create_group('metadata')
        meta_group.attrs['seed'] = seed
        meta_group.attrs['num_agvs'] = env.num_agvs
        meta_group.attrs['num_pickers'] = env.num_pickers
        meta_group.attrs['grid_size'] = env.grid_size
        
        # Rack locations 저장 (그래프 구축용)
        rack_locations = []
        for group_idx, group_positions in enumerate(env.rack_groups):
            for pos in group_positions:
                rack_locations.append([pos[0], pos[1], group_idx])
        meta_group.create_dataset('rack_locations', data=np.array(rack_locations))
        
        self.step_data_buffer = []
        
    def log_step_pre(self, env, actions, step_id):
        # 현재 환경 상태 추출
        step_data = {
            'step_id': step_id,
            'actions': np.array(actions),
            
            # Agent 상태
            'agent_positions': np.array([(a.x, a.y) for a in env.agents]),
            'agent_directions': np.array([a.dir.value for a in env.agents]),
            'agent_busy': np.array([a.busy for a in env.agents]),
            'agent_carrying_shelf': np.array([a.carrying_shelf is not None for a in env.agents]),
            'agent_targets': np.array([a.target for a in env.agents]),
            
            # 환경 상태
            'grid_collision_layers': env.grid.copy(),
            'request_queue_ids': np.array([shelf.id for shelf in env.request_queue]),
            
            # 그래프 구축을 위한 추가 정보
            'shelf_request_info': env.get_shelf_request_information(),
            'empty_shelf_info': env.get_empty_shelf_information(),
        }
        
        # 현재 observation도 저장 (그래프 변환용)
        try:
            current_obs = [env.observation_space_mapper.observation(agent) for agent in env.agents]
            step_data['observations'] = np.array(current_obs)
        except:
            # observation이 없는 경우 패스
            pass
            
        self.step_data_buffer.append(step_data)
        
    def log_step_post(self, rewards, info):
        # 마지막 step 데이터에 결과 추가
        if self.step_data_buffer:
            self.step_data_buffer[-1]['rewards'] = np.array(rewards)
            self.step_data_buffer[-1]['info'] = dict(info)
            
    def end_episode(self):
        if not self.current_episode_group or not self.step_data_buffer:
            return
            
        # Steps 그룹 생성
        steps_group = self.current_episode_group.create_group('steps')
        
        # 각 step 데이터를 HDF5에 저장
        for step_data in self.step_data_buffer:
            step_group = steps_group.create_group(f"step_{step_data['step_id']:06d}")
            
            # 각 데이터 저장
            for key, value in step_data.items():
                if key == 'step_id':
                    continue
                elif key == 'info':
                    # info 딕셔너리는 attributes로 저장
                    for info_key, info_value in value.items():
                        step_group.attrs[f'info_{info_key}'] = info_value
                else:
                    step_group.create_dataset(key, 
                                            data=value, 
                                            compression='gzip',
                                            compression_opts=1)
        # print(f"Episode ended with {len(self.step_data_buffer)} steps")
        # Episode 통계 저장
        summary_group = self.current_episode_group.create_group('summary')
        if self.step_data_buffer:
            total_rewards = sum(step_data.get('rewards', np.zeros(1)) for step_data in self.step_data_buffer)
            summary_group.create_dataset('episode_returns', data=total_rewards)
            summary_group.attrs['episode_length'] = len(self.step_data_buffer)
            
        self.step_data_buffer = []
        self.current_episode_group = None
        
    def close(self):
        if self.file:
            self.file.close()
            
    def __del__(self):
        self.close()

class LoggingWarehouseWrapper:
    def __init__(self, env, log_file_path):
        self.env = env
        self.unwrapped = env
        self.logger = HDF5Logger(log_file_path)
        self.episode_count = 0
        
    def reset(self, seed=None):
        obs = self.env.reset(seed=seed)
        self.logger.start_episode(self.episode_count, seed, self.env)
        self.step_count = 0
        return obs
        
    def step(self, actions):
        # 현재 상태 로깅 (step 실행 전)
        self.logger.log_step_pre(self.env, actions, self.step_count)
        
        # 원본 step 실행
        obs, rewards, terminated, truncated, info = self.env.step(actions)
        
        # 결과 로깅 (step 실행 후)
        self.logger.log_step_post(rewards, info)
        
        self.step_count += 1
        
        done = terminated or truncated
        if all(done):
            self.logger.end_episode()
            self.episode_count += 1
            
        return obs, rewards, terminated, truncated, info

    def render(self, mode="human"):
        return self.env.render(mode)

# Mission 클래스 정의 (heuristic.py에서 import 안 될 경우)
from dataclasses import dataclass
from enum import Enum

class MissionType(Enum):
    PICKING = 1
    RETURNING = 2
    DELIVERING = 3

@dataclass
class Mission:
    mission_type: MissionType
    location_id: int
    location_x: int
    location_y: int
    assigned_time: int
    at_location: bool = False

def heuristic_episode_with_logging(env, render=False, seed=None):
    """
    Modified heuristic episode that works with LoggingWarehouseWrapper
    """
    # non_goal_location_ids corresponds to the item ordering in `get_empty_shelf_information`
    non_goal_location_ids = []
    for id_, coords in env.unwrapped.action_id_to_coords_map.items():
        if (coords[1], coords[0]) not in env.unwrapped.goals:
            non_goal_location_ids.append(id_)
    non_goal_location_ids = np.array(non_goal_location_ids)
    location_map = env.unwrapped.action_id_to_coords_map
    
    # Reset through wrapper - this will trigger logging
    _ = env.reset(seed=seed)
    done = False
    all_infos = []
    timestep = 0

    agents = env.unwrapped.agents
    agvs = [a for a in agents if a.type == AgentType.AGV]
    pickers = [a for a in agents if a.type == AgentType.PICKER]
    coords_original_loc_map = {v:k for k, v in env.unwrapped.action_id_to_coords_map.items()}
    
    # split the pickers evenly into sections throughout the warehouse
    sections = env.unwrapped.rack_groups
    picker_sections = split_list(sections, len(pickers))
    picker_sections = [flatten_list(l) for l in picker_sections]

    assigned_agvs: dict = OrderedDict({}) 
    assigned_pickers: dict = OrderedDict({})
    assigned_items: dict = OrderedDict({})
    global_episode_return = 0
    episode_returns = np.zeros(env.unwrapped.num_agents)
    
    while not done:
        request_queue = env.unwrapped.request_queue
        goal_locations = env.unwrapped.goals
        actions = {k: 0 for k in agents}

        # [AGV None -> AGV PICKING] find closest non-busy agv agent to each item in request queue
        for item in request_queue:
            if item.id in assigned_items.values():
                continue

            available_agvs = [a for a in agvs if not a.busy and not a.carrying_shelf]
            available_agvs = [a for a in available_agvs if not a in assigned_agvs]

            if not available_agvs:
                continue

            agv_shortest_paths = [env.unwrapped.find_path((a.y, a.x), (item.y, item.x), a, care_for_agents=False) for a in available_agvs]
            agv_distances = [len(p) for p in agv_shortest_paths]
            closest_agv = available_agvs[np.argmin(agv_distances)]
            item_location_id = coords_original_loc_map[(item.y, item.x)]
            if closest_agv:
                assigned_agvs[closest_agv] = Mission(MissionType.PICKING, item_location_id, item.x, item.y, timestep)
                assigned_items[closest_agv] = item.id

        for agv in agvs:
            if agv in assigned_agvs and (agv.x == assigned_agvs[agv].location_x) and (agv.y == assigned_agvs[agv].location_y):
                assigned_agvs[agv].at_location = True

            if agv not in assigned_agvs or agv.busy:
                continue

            # [AGV PICKING -> AGV DELIVERING]
            if assigned_agvs[agv].mission_type == MissionType.PICKING and assigned_agvs[agv].at_location and agv.carrying_shelf:
                goal_shortest_paths = [env.unwrapped.find_path((agv.y, agv.x), (y, x), agv, care_for_agents=False) for (x,y) in goal_locations]
                goal_distances = [len(p) for p in goal_shortest_paths] 
                closest_goal = goal_locations[np.argmin(goal_distances)]
                goal_location_id = coords_original_loc_map[(closest_goal[1], closest_goal[0])]
                mission = assigned_agvs.pop(agv)
                assigned_agvs[agv] = Mission(MissionType.DELIVERING, goal_location_id, closest_goal[0], closest_goal[1], timestep)

            # [AGV DELIVERING -> AGV RETURNING]
            if assigned_agvs[agv].mission_type == MissionType.DELIVERING and assigned_agvs[agv].at_location and agv.carrying_shelf:
                empty_shelves = env.unwrapped.get_empty_shelf_information()
                empty_location_ids = list(non_goal_location_ids[empty_shelves > 0])
                assigned_item_loc_agvs = [mission.location_id for mission in assigned_agvs.values()]
                empty_location_ids = [loc_id for loc_id in empty_location_ids if loc_id not in assigned_item_loc_agvs]
                empty_location_yx = [location_map[i] for i in empty_location_ids]
                closest_empty_location_paths = [env.unwrapped.find_path((agv.y, agv.x), (y, x), agv, care_for_agents=False) for (y,x) in empty_location_yx]
                closest_empty_location_distances = [len(p) for p in closest_empty_location_paths]
                if len(closest_empty_location_distances) == 0:
                    closest_location_id = None
                else:
                    closest_location_id = empty_location_ids[np.argmin(closest_empty_location_distances)]
                closest_location_yx = location_map[closest_location_id]
                assigned_agvs.pop(agv)
                assigned_agvs[agv] = Mission(MissionType.RETURNING, closest_location_id, closest_location_yx[1], closest_location_yx[0], timestep)

            # [AGV RETURNING -> AGV None]
            if assigned_agvs[agv].mission_type == MissionType.RETURNING and assigned_agvs[agv].at_location and not agv.carrying_shelf:
                assigned_agvs.pop(agv)
                assigned_items.pop(agv)

        # Send pickers to where AGVs are going
        for agv, mission in assigned_agvs.items():
            if mission.mission_type in [MissionType.PICKING, MissionType.RETURNING]:
                in_pickers_zone = [(mission.location_y, mission.location_x) in p for p in picker_sections]
                relevant_picker = pickers[in_pickers_zone.index(True)]
                if relevant_picker not in assigned_pickers.keys():
                    assigned_pickers[relevant_picker] = Mission(MissionType.PICKING, mission.location_id, mission.location_x, mission.location_y, timestep)

        # Picker has reached destination, remove its mission
        for picker in pickers:
            if picker in assigned_pickers and (picker.x == assigned_pickers[picker].location_x) and (picker.y == assigned_pickers[picker].location_y):
                assigned_pickers[picker].at_location = True
                assigned_pickers.pop(picker)

        # Map the missions to actions
        for agv, mission in assigned_agvs.items():
            actions[agv] = mission.location_id if not agv.busy else 0
        for picker, mission in assigned_pickers.items():
            actions[picker] = mission.location_id

        if render:
            env.render(mode="human")

        # Step through wrapper - this will trigger logging
        _, reward, terminated, truncated, info = env.step(list(actions.values()))
        done = terminated or truncated
        episode_returns += np.array(reward, dtype=np.float64)
        global_episode_return += np.sum(reward)
        done = all(done)
        all_infos.append(info)
        timestep += 1

    return all_infos, global_episode_return, episode_returns
   
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

def collect_data(env_id, num_episodes, seed):
    base_env = gym.make(env_id)
    env = LoggingWarehouseWrapper(base_env.unwrapped, f"warehouse_data_{env_id}_seed{seed}.h5")
    completed_episodes = 0
    
    for i in range(num_episodes):
        start = time.time()
        # 수정: wrapper를 통해 episode 실행
        infos, global_episode_return, episode_returns = heuristic_episode_with_logging(
            env, False, seed+i
        )
        end = time.time()
        
        last_info = info_statistics(infos, global_episode_return, episode_returns)
        last_info["overall_pick_rate"] = last_info.get("total_deliveries") * 3600 / (5 * last_info['episode_length'])
        episode_length = len(infos)
        print(f"Env: {env_id} | Seed: {seed} | Episode {completed_episodes}: | [Overall Pick Rate={last_info.get('overall_pick_rate'):.2f}]| [Global return={last_info.get('global_episode_return'):.2f}]| [Total shelf deliveries={last_info.get('total_deliveries'):.2f}]| [Total clashes={last_info.get('total_clashes'):.2f}]| [Total stuck={last_info.get('total_stuck'):.2f}] | [FPS = {episode_length/(end-start):.2f}]")
        completed_episodes += 1
    
    # 로거 종료
    env.logger.close()

if __name__ == "__main__":
    # environments = [
    #     'tarware-tiny-3agvs-2pickers-partialobs-v1',
    #     'tarware-small-6agvs-3pickers-partialobs-v1', 
    #     'tarware-medium-10agvs-5pickers-partialobs-v1',
    #     'tarware-medium-19agvs-9pickers-partialobs-v1',
    #     'tarware-large-15agvs-8pickers-partialobs-v1'
    # ]
    environments = [
        'tarware-small-6agvs-3pickers-partialobs-v1', 
        'tarware-medium-10agvs-5pickers-partialobs-v1',
        'tarware-medium-19agvs-9pickers-partialobs-v1',
        'tarware-large-15agvs-8pickers-partialobs-v1'
    ]
    # environments = [
    #     'tarware-tiny-3agvs-2pickers-partialobs-v1']
    base_seeds = [0, 1000, 2000, 3000, 4000]
    # base_seeds = [0]
    for env_id in environments:
        for seed in base_seeds:
            print(f"Starting data collection for {env_id} with seed {seed}")
            collect_data(env_id, num_episodes=200, seed=seed)
            # collect_data(env_id, num_episodes=1, seed=seed)

            print(f"Completed data collection for {env_id} with seed {seed}")