from .MultiAgentGlobalObservationSpace import MultiAgentGlobalObservationSpace
from .MultiAgentPartialObservationSpace import \
    MultiAgentPartialObservationSpace
from .MultiAgentGraphObservationSpace import MultiAgentGraphObservationSpace
observation_map = {
    'partial': MultiAgentPartialObservationSpace,
    'global': MultiAgentGlobalObservationSpace
}