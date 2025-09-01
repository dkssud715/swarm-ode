import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import gymnasium as gym

from tarware.heuristic import heuristic_episode
import h5py
import numpy as np
from collections import defaultdict

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
        
        if terminated or truncated:
            self.logger.end_episode()
            self.episode_count += 1
            
        return obs, rewards, terminated, truncated, info

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
    env = gym.make(env_id)
    env = LoggingWarehouseWrapper(env.unwrapped, f"warehoues_data_seed{seed}.h5")
    completed_episodes = 0
    for i in range(num_episodes):
        start = time.time()
        infos, global_episode_return, episode_returns = heuristic_episode(env.unwrapped, args.render, seed+i)
        end = time.time()
        last_info = info_statistics(infos, global_episode_return, episode_returns)
        last_info["overall_pick_rate"] = last_info.get("total_deliveries") * 3600 / (5 * last_info['episode_length'])
        episode_length = len(infos)
        print(f"Completed Episode {completed_episodes}: | [Overall Pick Rate={last_info.get('overall_pick_rate'):.2f}]| [Global return={last_info.get('global_episode_return'):.2f}]| [Total shelf deliveries={last_info.get('total_deliveries'):.2f}]| [Total clashes={last_info.get('total_clashes'):.2f}]| [Total stuck={last_info.get('total_stuck'):.2f}] | [FPS = {episode_length/(end-start):.2f}]")
        completed_episodes += 1

if __name__ == "__main__":
    environments = [
    'tarware-tiny-3agvs-2pickers-partialobs-v1',
    'tarware-small-6agvs-3pickers-partialobs-v1', 
    'tarware-medium-10agvs-5pickers-partialobs-v1',
    'tarware-medium-19agvs-9pickers-partialobs-v1',
    'tarware-large-15agvs-8pickers-partialobs-v1'
    ]

    base_seeds = [0, 1000, 2000, 3000, 4000]

    for env_id in environments:
        for seed in base_seeds:
            # 각 조합마다 200 episodes 수집
            collect_data(env_id, episodes=200, seed=seed)
