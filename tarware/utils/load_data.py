def load_episode_data(filepath, episode_id):
    """저장된 에피소드 데이터를 로딩하여 그래프 변환기에서 사용할 수 있는 형태로 변환"""
    with h5py.File(filepath, 'r') as f:
        ep_group = f[f'episode_{episode_id:06d}']
        
        # 메타데이터 로딩
        metadata = dict(ep_group['metadata'].attrs)
        rack_locations = ep_group['metadata/rack_locations'][:]
        
        # 모든 스텝 데이터 로딩
        episodes_data = []
        steps_group = ep_group['steps']
        
        for step_name in sorted(steps_group.keys()):
            step_group = steps_group[step_name]
            step_data = {}
            
            # 데이터셋 로딩
            for key in step_group.keys():
                step_data[key] = step_group[key][:]
                
            # Attributes 로딩
            for attr_key, attr_value in step_group.attrs.items():
                step_data[attr_key] = attr_value
                
            episodes_data.append(step_data)
            
        return metadata, rack_locations, episodes_data

# 그래프 변환 예시
def convert_logged_data_to_graph(filepath, episode_id, converter):
    metadata, rack_locations, episode_data = load_episode_data(filepath, episode_id)
    
    graphs = []
    for step_data in episode_data:
        if 'observations' in step_data:
            graph = converter._build_graph_from_observation(
                step_data['observations'], 
                [(x, y, g) for x, y, g in rack_locations]
            )
            graphs.append(graph)
    
    return graphs