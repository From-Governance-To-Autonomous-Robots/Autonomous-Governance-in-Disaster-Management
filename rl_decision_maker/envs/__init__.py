from gym.envs.registration import register
from .taskEnvTrain import TaskSequenceTrainEnv
from .taskEnvVal import TaskSequenceValEnv

register(
    id='TaskSequenceTrainEnv-v0',
    entry_point='envs.taskEnvTrain:TaskSequenceTrainEnv',
)

register(
    id='TaskSequenceValEnv-v0',
    entry_point='envs.taskEnvVal:TaskSequenceValEnv',
)
